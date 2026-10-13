"""Bounded LRU of image embeddings shared by the model servers.

A vision embedding takes fresh pseudo token ids every time it is built, and the paged KV cache keys its pages on
token ids. Building the embedding again for the same picture therefore breaks the prefix match at that picture, on
every later request. This cache returns the very same embedding object (same ids, same text alias) for the same
picture, and it skips the vision tower for a picture it already holds.

One picture used n times in a prompt needs n distinct embedding objects (an object spliced twice would make its
ids ambiguous), so the key is (content hash, occurrence in the prompt); each copy stays stable across requests.

Memory: one entry holds the embedding rows on the host, tokens x hidden x 2 bytes (half), plus the same again for
each deepstack layer. A 1280x800 screenshot is about 1000 rows: 10 MB at hidden 5120, about 40 MB with three
deepstack layers. The cache is bounded by entry count and by total bytes.
"""
from __future__ import annotations

import collections
import hashlib
import threading
from typing import Any, Callable

DEFAULT_MAX_ITEMS = 32           # a long computer-use session re-sends every earlier screenshot
DEFAULT_MAX_BYTES = 1 << 30      # 1 GiB of host memory


def digest(raw: bytes) -> str:
    """Content key of an encoded image. One server process holds one vision tower with fixed resize and pixel limits,
    so the bytes alone decide the embedding; the cache dies with the process."""
    return hashlib.sha256(raw).hexdigest()


def embedding_nbytes(emb: Any) -> int:
    """Host bytes of an MMEmbedding: its rows and its deepstack rows."""
    total = 0
    for t in [getattr(emb, "embeddings", None), *(getattr(emb, "deepstack_embeddings", None) or [])]:
        if t is not None:
            total += int(t.numel()) * int(t.element_size())
    return total


class ImageEmbeddingCache:
    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS, max_bytes: int = DEFAULT_MAX_BYTES):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items: collections.OrderedDict = collections.OrderedDict()   # (key, occurrence) -> (embedding, nbytes)
        self._nbytes = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._nbytes = 0

    def get_many(self, images: list[tuple[Any, str]], build: Callable[[Any], Any]) -> list:
        """One embedding per (image, key), in order. `build(image)` runs only for a picture not held yet.
        The lock stays held while building, so two requests never encode the same picture twice."""
        out, seen = [], collections.Counter()
        with self._lock:
            for image, key in images:
                slot = (key, seen[key])
                seen[key] += 1
                hit = self._items.get(slot)
                if hit is None:
                    emb = build(image)
                    n = embedding_nbytes(emb)
                    self._items[slot] = (emb, n)
                    self._nbytes += n
                else:
                    emb = hit[0]
                self._items.move_to_end(slot)
                out.append(emb)
            while self._items and (len(self._items) > self.max_items or self._nbytes > self.max_bytes):
                _, (_, n) = self._items.popitem(last=False)
                self._nbytes -= n
        return out
