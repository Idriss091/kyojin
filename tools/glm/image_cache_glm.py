#!/usr/bin/env python3
"""Bounded cache of image embeddings for the GLM server: a full cache gives up the *tail* of the prompt.

A vision embedding takes fresh pseudo token ids every time it is built, and the paged KV cache keys its pages on
token ids. Building the embedding again for the same picture therefore breaks the prefix match at that picture, on
every later request. This cache returns the very same embedding object (same ids, same text alias) for the same
picture, and it skips the vision tower for a picture it already holds.

One picture used n times in a prompt needs n distinct embedding objects (an object spliced twice would make its
ids ambiguous), so the key is (content hash, occurrence in the prompt); each copy stays stable across requests.

What a hit is worth depends on *where* the picture sits in the prompt: the paged prefix matches from the first
token on and stops at the first picture that had to be re-encoded, so the n-th picture buys reuse only if every
earlier one is still held, and the first picture buys the whole prompt. The eviction order therefore follows the
prompt, not the clock: an entry no longer referenced by a remembered prompt goes first, then the *last* pictures
of the remembered prompts, and the head of the prompt is the last thing given up. A plain LRU does the opposite —
it drops the oldest pictures, which are exactly the ones the match depends on, and turns "the cache is one picture
too small" into "nothing at all is reused" (measured: 21 pictures against a 20-picture budget, 99.8 % of a 63K
prompt reused before the change, 0 % — the full 63K re-prefilled — after it).

The cache is bounded by total bytes, with an optional entry count on top; the GLM server sizes the byte budget
from RAM and its own flags (see `--image-cache-mb`). Each entry holds the embedding rows on the host plus the same
again for each deepstack layer, so the budget is host memory, not VRAM: GLM's vision merger emits fp32, about
49.7 MiB for one 3179-token manga page and 1 MB for a 224x224 crop.

When the budget cannot hold every active conversation, each one keeps its head and re-encodes its tail: the loss
is a few pictures per turn, not the prompt. `max_orders` is how many recent prompts the policy weighs, and among
pictures at the same distance from the start, the one from the conversation served longest ago goes first.
"""
from __future__ import annotations

import collections
import hashlib
import threading
from typing import Any, Callable

DEFAULT_MAX_ITEMS = 0            # the byte budget is the real bound; a count cap is optional on top of it
DEFAULT_MAX_BYTES = 2 << 30      # 2 GiB of host memory (Strix Halo: ~104 GB of weights + the KV pool)
DEFAULT_MAX_ORDERS = 2           # prompts whose picture order the eviction policy remembers


def digest(raw: bytes) -> str:
    """Content key of an encoded image. One server process holds one vision tower with fixed resize and pixel limits,
    so the bytes alone decide the embedding; the cache dies with the process."""
    return hashlib.sha256(raw).hexdigest()


def embedding_nbytes(emb: Any) -> int:
    """Host bytes of an MMEmbedding: its rows and its deepstack rows, in whatever form they are stored.
    Quantized rows report their own size (int8 plus scales), so the budget counts what is really held."""
    total = 0
    for t in [getattr(emb, "embeddings", None), *(getattr(emb, "deepstack_embeddings", None) or [])]:
        if t is None:
            continue
        nbytes = getattr(t, "nbytes", None)
        total += int(nbytes) if nbytes is not None else int(t.numel()) * int(t.element_size())
    return total


class ImageEmbeddingCache:
    """Embeddings keyed by (content, occurrence), sized in bytes, evicted from the end of the prompt backwards.

    `max_bytes` or `max_items` at 0 means that bound is not applied. `max_orders` is how many recent prompts the
    eviction policy keeps in mind: 1 is enough for one request at a time and gives exactly "give up the tail of
    this prompt", while a server running `--sessions N` passes N (or a little more) so that concurrent
    conversations do not evict each other's heads. Thread safe: one vision pass per picture, also across requests.
    """

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS, max_bytes: int = DEFAULT_MAX_BYTES,
                 max_orders: int = DEFAULT_MAX_ORDERS):
        self.max_items = max(0, int(max_items))
        self.max_bytes = max(0, int(max_bytes))
        self.max_orders = max(1, int(max_orders))
        self._items: collections.OrderedDict = collections.OrderedDict()   # slot -> (embedding, nbytes)
        self._orders: collections.OrderedDict = collections.OrderedDict()  # prompt serial -> [slot, ...]
        self._seq = 0
        self._nbytes = 0
        self._last_order: list = []          # slots of the prompt being served, in prompt order
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def live_orders(self) -> list:
        """The pictures the remembered prompts refer to, newest prompt first, as a list of content-key sets.

        A second-level store needs this to tell conversations apart: the newest entry is the one being served
        (its pictures must stay complete or the prompt cannot be reproduced), older entries are conversations the
        user may come back to, and a picture in none of them belongs to a conversation that is over. The cache
        only remembers the last `max_orders` prompts, so a conversation the user has left ages out on its own.
        """
        return [{slot[0] for slot in slots} for slots in reversed(self._orders.values())]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._orders.clear()
            self._nbytes = 0

    def _over_budget(self) -> bool:
        return bool((self.max_items and len(self._items) > self.max_items) or
                    (self.max_bytes and self._nbytes > self.max_bytes))

    def _eviction_order(self) -> list:
        """Slots of the held embeddings, the one to give up first first.

        Least valuable first: an entry that no remembered prompt refers to, then the entries of those prompts
        ordered by how far into the prompt they sit (the last picture of a prompt is worth the least), and among
        equal positions the prompt that was seen longest ago.
        """
        position: dict = {}
        for age, slots in enumerate(self._orders.values()):
            for index, slot in enumerate(slots):
                seen = position.get(slot)
                if seen is None or index < seen[0]:    # keep the earliest position this picture has in any prompt
                    position[slot] = (index, age)
        stale, live = [], []
        for lru, slot in enumerate(self._items):       # _items is in use order, least recently used first
            if slot in position:
                index, age = position[slot]
                live.append((-index, age, lru, slot))
            else:
                stale.append((lru, slot))
        stale.sort()
        live.sort()
        return [slot for _, slot in stale] + [slot for *_, slot in live]

    def _trim(self, evicted: list | None = None) -> int:
        """Evict until both bounds hold. Returns how many entries went.

        `evicted`, when given, collects the (key, embedding, position-in-this-prompt) of everything that left, so
        a caller can put them somewhere cheaper than memory (the server writes them to its picture cache on
        disk). Positions are the entries' ranks in the prompt being served, which is the order any second-level
        store has to keep: the earliest pictures are the ones the paged prefix needs.
        """
        if not self._over_budget():
            return 0
        rank = {slot: i for i, slot in enumerate(self._last_order or [])} if evicted is not None else {}
        gone = 0
        for slot in self._eviction_order():
            if not self._over_budget():
                break
            item = self._items.pop(slot, None)
            if item is not None:
                self._nbytes -= item[1]
                gone += 1
                if evicted is not None:
                    evicted.append((slot[0], item[0], rank.get(slot)))
        return gone

    def get_many(self, images: list[tuple[Any, str]], build: Callable[[Any], Any],
                 evicted: list | None = None, fetch: Callable[[Any, str], Any] | None = None,
                 mine: set | None = None) -> list:
        """One embedding per (image, key), in order. For a picture not held yet, `fetch(image, key)` is tried
        first (a second-level store, e.g. the one on disk) and `build(image)` runs only when it has nothing.
        The lock stays held while both run, so two requests never encode the same picture twice.

        The prompt order of this call is remembered for the next `max_orders` calls: they are what the eviction
        policy weighs, so a conversation that keeps sending its whole history keeps its earliest pictures.

        `evicted`, when given, is filled with what this call pushed out (see `_trim`), and `mine` with the
        content keys of *this* prompt: a caller that writes the evicted pictures somewhere slower needs to know
        which conversation it is serving, or concurrent requests would evict each other's caches. The caller must
        not touch the cache from a `build` callback and does its own work with both after this returns.
        """
        out, seen = [], collections.Counter()
        order: list = []
        with self._lock:
            for image, key in images:
                slot = (key, seen[key])
                seen[key] += 1
                order.append(slot)
                if mine is not None:
                    mine.add(key)
                hit = self._items.get(slot)
                if hit is None:
                    emb = fetch(image, key) if fetch is not None else None
                    if emb is None:
                        emb = build(image)
                    n = embedding_nbytes(emb)
                    self._items[slot] = (emb, n)
                    self._nbytes += n
                else:
                    emb = hit[0]
                self._items.move_to_end(slot)
                out.append(emb)
            if order:
                self._seq += 1
                self._orders[self._seq] = order
                while len(self._orders) > self.max_orders:
                    self._orders.popitem(last=False)
            self._last_order = order
            self._trim(evicted)
        return out
