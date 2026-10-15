#!/usr/bin/env python3
"""Local additions to tools/glm/test_serve.py: the request log and the image cache budget.

These tests cover work that is not upstream (yet), so they live in a file of their own instead of patching
upstream's test_serve.py: a merge of upstream's file never conflicts with them, and they can be folded back
into test_serve.py in one move if the features are ever proposed upstream. The fake engine below is a copy of
the one in test_serve.py, deliberately: this file then runs on its own and does not break when upstream edits
its fixtures.

    kyojin/.venv/bin/python -m unittest tools.glm.test_serve_glm_additional       # 11 tests, no GPU, no model
    kyojin/.venv/bin/python -m pytest -q tools/glm/test_serve_glm_additional.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve  # noqa: E402

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}: {{ m.content }}<|im_end|>{% endfor %}"
            "{% if tools %}TOOLS={{ tools|length }}{% endif %}")
# The same template with the image placeholders the server matches against the request's image parts.
IMAGE_TEMPLATE = TEMPLATE.replace(
    "{{ m.content }}",
    "{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}"
    "{% if p.type == 'text' %}{{ p.text }}{% else %}" + serve.IMAGE_TRIPLE + "{% endif %}{% endfor %}{% endif %}")


class FakeGenerator:
    num_draft_tokens = 2  # /metrics derives the round count from the engine's window


class FakeEngine:
    def __init__(self, output: str):
        self.output = output
        self.prompts = []
        self.greedy_generator = FakeGenerator()

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.last_stats = {"cached_tokens": 8, "new_tokens": 4, "prompt_tokens": 20,
                           "time_prefill": 0.5, "time_generate": 1.0,
                           "accepted_draft_tokens": 6, "rejected_draft_tokens": 6}
        for i in range(0, len(self.output), 3):  # small chunks cut tags in half
            yield self.output[i:i + 3]


def run(coro):
    return asyncio.run(coro)


def vision_engine():
    """A fake engine whose generate() runs the real _embed() on a real cache, plus a tower that counts encodes."""
    import itertools

    import torch
    from image_cache import ImageEmbeddingCache

    ids, calls = itertools.count(1_000_000_000, 100), []

    class Emb:
        def __init__(self):
            self.first_index = next(ids)
            self.text_alias = f"<$EMB_{self.first_index}$>"
            self.mm_length = 3
            self.embeddings = torch.zeros(3, 64, dtype=torch.half)
            self.deepstack_embeddings = None

    class Tower:
        def get_image_embeddings(self, *args, **kw):
            calls.append(1)
            return Emb()

    eng = object.__new__(serve.ResidentEngine)
    eng.emb_cache, eng.vision, eng.tokenizer = ImageEmbeddingCache(), Tower(), None
    return eng, calls


def disk_engine(directory, budget_bytes: int = 768, mode: str = "fp32", cap_bytes: int = 0, on_disk: bool = True,
                lock_s: float = 0.0, policy: str = "tail"):
    """An engine whose _embed() runs over a real memory cache *and* a real second-level cache on disk.

    The memory cache runs the `tail` policy the disk layer is built for: keeping the head of a prompt is what
    lets a restored picture complete it. The server's own default is the shared cache's LRU (`head`).

    The fake pictures carry everything a restore needs (identity, token list, MRoPE metadata) so a test can ask
    whether the prompt would be reproduced: unlike the tower, the disk has to give back the same pseudo ids.
    """
    import itertools

    import torch
    from image_cache import ImageEmbeddingCache

    ids, calls = itertools.count(1_000_000_000, 100), []

    class Emb:
        def __init__(self):
            self.first_index = next(ids)
            self.mm_length = 3
            self.last_index = self.first_index + self.mm_length
            self.embeddings = torch.zeros(3, 64, dtype=torch.half)
            self.deepstack_embeddings = None
            self.grid_thw = (1, 1, 1)
            self.mrope_merge_size = 2
            self.metadata = {"original_size": (8, 8)}
            self.token_string = torch.tensor([[7] + list(range(self.first_index, self.last_index)) + [8]])
            self.token_list = self.token_string[0].tolist()
            self.text_alias = f"<$EMB_{self.first_index}$>"

    class Tower:
        def get_image_embeddings(self, *args, **kw):
            calls.append(1)
            return Emb()

    eng = object.__new__(serve.ResidentEngine)
    eng.emb_cache = ImageEmbeddingCache(max_bytes=budget_bytes, policy=policy)
    eng.vision, eng.tokenizer = Tower(), None
    eng.image_cache_format = mode
    eng.emb_disk = None
    if on_disk:
        from image_cache_disk import ImageEmbeddingDisk
        eng.emb_disk = ImageEmbeddingDisk(directory, max_bytes=cap_bytes, start_token=7, end_token=8, lock_s=lock_s)
    return eng, calls


def tiny_pictures(count: int):
    """Whatever `_embed` needs: a PIL image and its content key."""
    from PIL import Image
    return [(Image.new("RGB", (8, 8), (i * 30 % 255, 0, 0)), f"key{i}") for i in range(count)]


class ImageDiskCacheTests(unittest.TestCase):
    """--image-cache-dir: what leaves memory is written once, and comes back with its identity."""

    def test_the_disk_cache_is_a_flag_not_an_environment_variable(self):
        self.assertNotIn("EXL3_IMAGE_CACHE_DIR", os.environ)
        self.assertNotIn("EXL3_IMAGE_CACHE_MB", os.environ)

    def test_the_tenure_reaches_the_disk_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, _ = disk_engine(tmp, budget_bytes=384, lock_s=42.0)
            self.assertEqual(eng.emb_disk.lock_s, 42.0)
            eng, _ = disk_engine(tmp, budget_bytes=384, lock_s=0.0)
            self.assertEqual(eng.emb_disk.lock_s, 0.0)

    def test_only_what_memory_gave_up_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768)        # room for two of the four pictures
            images = tiny_pictures(4)
            serve.ResidentEngine._embed(eng, images)
            self.assertEqual(len(calls), 4)                        # a cold pass encodes all four
            self.assertEqual(len(list(pathlib.Path(tmp).glob("*.pt"))), 2)   # ... and two were evicted
            self.assertEqual(eng.emb_disk.report()["stored"], 2)

            serve.ResidentEngine._embed(eng, images)               # same conversation again
            self.assertEqual(len(calls), 4)                        # two from memory, two from disk: no vision pass
            self.assertEqual(eng.emb_disk.report()["loaded"], 2)

    def test_a_restored_picture_keeps_the_ids_the_prompt_was_built_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=384)        # room for one
            images = tiny_pictures(2)
            first = serve.ResidentEngine._embed(eng, images)
            self.assertEqual(len(list(pathlib.Path(tmp).glob("*.pt"))), 1)
            self.assertEqual(first[1].embeddings.shape, (3, 64))   # the second picture went to disk

            again = serve.ResidentEngine._embed(eng, images)
            self.assertEqual(len(calls), 2)                        # no second vision pass
            self.assertEqual(again[1].first_index, first[1].first_index)
            self.assertEqual(again[1].token_list, first[1].token_list)
            self.assertEqual(again[1].text_alias, first[1].text_alias)
            self.assertEqual(again[1].grid_thw, first[1].grid_thw)

    def test_without_a_directory_nothing_is_written_and_the_tower_runs_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768, on_disk=False)
            images = tiny_pictures(4)
            serve.ResidentEngine._embed(eng, images)
            serve.ResidentEngine._embed(eng, images)
            self.assertEqual(list(pathlib.Path(tmp).rglob("*.pt")), [])
            self.assertEqual(len(calls), 6)                        # 4 cold, then the 2 that memory dropped

    def test_a_batch_that_does_not_fit_writes_nothing_at_all(self):
        # What matters for wear: a conversation that cannot be cached should cost *no* bytes at all, rather than
        # being written and deleted again on the next request.
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768, cap_bytes=1)     # room for nothing
            serve.ResidentEngine._embed(eng, tiny_pictures(4))
            report = eng.emb_disk.report()
            self.assertEqual(report["files"], 0)
            self.assertEqual(report["written_bytes"], 0)
            self.assertEqual(report["dropped"], 0)
            self.assertGreater(report["skipped"], 0)
            self.assertEqual(list(pathlib.Path(tmp).glob("*.pt")), [])

    def test_every_picture_is_written_once_when_it_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng, calls = disk_engine(tmp, budget_bytes=768, cap_bytes=1 << 20)
            serve.ResidentEngine._embed(eng, tiny_pictures(4))
            first = eng.emb_disk.report()
            self.assertEqual((first["stored"], first["dropped"], first["skipped"]), (2, 0, 0))
            serve.ResidentEngine._embed(eng, tiny_pictures(4))               # the same two are evicted again
            again = eng.emb_disk.report()
            self.assertEqual(again["stored"], 2)                            # already on disk: not rewritten
            self.assertEqual(again["written_bytes"], first["written_bytes"])
            self.assertEqual(again["dropped"], 0)

    def test_an_unusable_directory_does_not_stop_a_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = pathlib.Path(tmp) / "blocker"
            blocker.write_text("not a directory")
            eng, calls = disk_engine(str(blocker), budget_bytes=384)
            self.assertFalse(eng.emb_disk.enabled)
            out = serve.ResidentEngine._embed(eng, tiny_pictures(2))
            self.assertEqual(len(out), 2)
            self.assertEqual(len(calls), 2)

    def test_the_defaults_match_upstream(self):
        # The memory budget is upstream's 1 GiB, and the disk cache is off unless a cap is given: no auto share
        # of RAM, and no way to ask for an unbounded directory.
        from image_cache import DEFAULT_MAX_ITEMS
        self.assertEqual(serve.IMAGE_CACHE_MB_DEFAULT, 1024)
        self.assertEqual(serve.make_image_cache(None).max_bytes, 1024 << 20)
        # The count cap defaults to the shared cache's, so the three servers behave the same out of the box.
        self.assertEqual(serve.IMAGE_CACHE_ITEMS_DEFAULT, DEFAULT_MAX_ITEMS)
        self.assertEqual(serve.make_image_cache(None).max_items, DEFAULT_MAX_ITEMS)
        self.assertEqual(serve.make_image_cache(None, 0).max_items, 0)          # 0 = no limit, when asked for
        self.assertEqual(serve.IMAGE_CACHE_DIR_MAX_GB, 0.0)
        self.assertTrue(serve.IMAGE_CACHE_DIR_DEFAULT.endswith(".cache/kyojin/image-embeddings"))
        self.assertEqual(serve.IMAGE_CACHE_DIR_LOCK_S, 300.0)

    def test_the_disk_cache_needs_a_cap(self):
        # `--image-cache-dir-max-gb 0` (the default) is off; any positive value is a real cap, never "unlimited".
        self.assertIsNone(serve.make_image_disk(serve.IMAGE_CACHE_DIR_DEFAULT, 0, 300.0))
        self.assertIsNone(serve.make_image_disk(None, 0.0, 300.0))
        with tempfile.TemporaryDirectory() as tmp:
            disk = serve.make_image_disk(tmp, 0.5, 60.0, start_token=7, end_token=8)
            self.assertEqual(disk.max_bytes, int(0.5 * 2 ** 30))
            self.assertEqual(disk.lock_s, 60.0)

    def test_flags_reach_the_cache(self):
        cache = serve.make_image_cache(1024 << 20, image_cache_items=7, sessions=3)
        self.assertEqual(cache.max_bytes, 1024 << 20)
        self.assertEqual(cache.max_items, 7)
        self.assertEqual(cache.max_orders, 3)                # one prompt order per session
        self.assertEqual(serve.make_image_cache(512 << 20, sessions=1).max_orders, 2)
        self.assertEqual(serve.make_image_cache(0, 0).max_bytes, 0)              # 0 = no bound
        self.assertEqual(serve.make_image_cache(0, 0).max_items, 0)
        self.assertEqual(serve.make_image_cache().max_bytes, 1024 << 20)         # engine default: 1 GiB

    def test_the_cache_the_engine_would_build_reports_its_budget(self):
        cache = serve.make_image_cache(serve.IMAGE_CACHE_MB_DEFAULT << 20, 0)
        self.assertEqual(cache.max_bytes, 1024 << 20)
        self.assertEqual(cache.max_items, 0)

    def test_the_cli_format_list_covers_every_engine_mode(self):
        from exllamav3.tokenizer.mm_embedding import MM_QUANT_MODES
        self.assertEqual(set(serve.IMAGE_CACHE_FORMATS), {"fp32", "fp16"} | set(MM_QUANT_MODES))

    def test_embed_stores_the_rows_in_the_requested_format(self):
        # --image-cache-format: fp32 and fp16 leave a plain tensor, q8 keeps the rows 8 bit. Either way the
        # object, its pseudo token ids and its length are the ones the prompt cache is keyed on, and the cache
        # counts the bytes of the form it actually holds (3 rows x 64 half values = 384 B as they are).
        from PIL import Image
        from exllamav3.tokenizer.mm_embedding import QuantizedMMRows
        from image_cache import embedding_nbytes
        eng, _ = vision_engine()
        img = Image.new("RGB", (8, 8))
        counts, packed = {}, {}
        for mode in serve.IMAGE_CACHE_FORMATS:
            eng.image_cache_format = mode
            eng.emb_cache.clear()
            emb = serve.ResidentEngine._embed(eng, [(img, f"key-{mode}")], None)[0]
            self.assertEqual(emb.mm_length, 3)
            self.assertEqual(emb.text_alias, f"<$EMB_{emb.first_index}$>")    # ids and alias untouched
            self.assertEqual(int(emb.embeddings.numel()), 192)                # logical values, not packed bytes
            packed[mode] = isinstance(emb.embeddings, QuantizedMMRows)
            counts[mode] = embedding_nbytes(emb)
        self.assertEqual(packed, {"fp32": False, "fp16": False, "q8": True})
        self.assertEqual(counts["fp32"], 384)                                  # as the tower emitted them
        self.assertEqual(counts["fp16"], 384)                                  # two bytes a value
        self.assertEqual(counts["q8"], 192 + 3 * 2)                            # one byte each + one scale per row


class RequestLogTests(unittest.TestCase):
    """KYOJIN_REQLOG: off unless asked for; when on, one line per lifecycle event, numbers and names only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_path = Path(self.tmp.name) / "requests.log"
        for key in ("KYOJIN_REQLOG", "KYOJIN_REQLOG_HEARTBEAT_S"):
            os.environ.pop(key, None)
            self.addCleanup(os.environ.pop, key, None)

    def lines(self) -> list[str]:
        return self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []

    def events(self) -> list[tuple[str, dict]]:
        """(event name, fields) per line; the timestamp is field 0 and the event name field 1."""
        out = []
        for line in self.lines():
            parts = line.split(" ")
            fields = {}
            for item in parts[2:]:
                key, _, value = item.partition("=")
                fields[key] = value.strip('"')
            out.append((parts[1], fields))
        return out

    def names(self) -> list[str]:
        return [name for name, _ in self.events()]

    def test_off_by_default_and_a_request_writes_nothing(self):
        from serve_reqlog import ReqLog
        log = ReqLog.from_env()
        self.assertFalse(log.enabled)
        self.assertIsNone(log.path)
        self.assertIn("request log off", log.describe())
        log.log("req_start", req="r1")               # disabled: a no-op, never an error
        self.assertEqual(log.lines(), [])

        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            app = serve.create_app(FakeEngine("hello"), "m", TEMPLATE)
            self.assertFalse(app["reqlog"].enabled)
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}]})
            status = response.status
            await response.read()
            await client.close()
            return status

        self.assertEqual(run(check()), 200)
        self.assertFalse(self.log_path.exists())      # nothing is created, nothing is written
        self.assertFalse(list(Path(self.tmp.name).iterdir()))

    def test_lifecycle_lines_share_one_id_and_never_hold_request_text(self):
        os.environ["KYOJIN_REQLOG"] = str(self.log_path)
        secret, reply = "PROMPT-SECRET-7f3a", "REPLY-SECRET-9b1c"
        engine = FakeEngine(reply)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            app = serve.create_app(engine, "m", TEMPLATE)
            self.assertTrue(app["reqlog"].enabled)
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": secret}]})
            body = await response.json()
            await client.close()
            return body

        body = run(check())
        self.assertEqual(body["choices"][0]["message"]["content"], reply)
        events = dict(self.events())
        for want in ("req_start", "req_queued", "job_start", "job_first_token", "job_done", "req_done"):
            self.assertIn(want, events)
        ids = {fields["req"] for name, fields in self.events() if "req" in fields}
        self.assertEqual(len(ids), 1)                       # every line of the request carries the same id
        self.assertEqual(events["req_queued"]["queued_ahead"], "0")
        self.assertEqual(events["job_done"]["cache_n"], "8")   # the engine's own cached_tokens
        self.assertEqual(events["job_done"]["cache_pct"], "40")
        self.assertIn(secret, " ".join(engine.prompts))     # the prompt really did carry the secret
        blob = "\n".join(self.lines())
        self.assertNotIn(secret, blob)
        self.assertNotIn(reply, blob)

    def test_stream_headers_first_content_and_done(self):
        os.environ["KYOJIN_REQLOG"] = str(self.log_path)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            await response.read()
            await client.close()

        run(check())
        names = self.names()
        for want in ("sse_headers", "sse_first_content", "sse_done", "req_done"):
            self.assertIn(want, names)
        self.assertLess(names.index("sse_headers"), names.index("sse_first_content"))
        self.assertLess(names.index("sse_first_content"), names.index("sse_done"))

    def test_refusals_are_logged_without_quoting_the_body(self):
        os.environ["KYOJIN_REQLOG"] = str(self.log_path)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("hello"), "m", TEMPLATE)))
            await client.start_server()
            bad = await client.post("/v1/chat/completions", json={"model": "m"})          # no messages: 400
            broken = await client.post("/v1/chat/completions", data=b'{"model": "m", ',
                                       headers={"Content-Type": "application/json"})       # malformed JSON: 400
            await client.close()
            return bad.status, broken.status

        self.assertEqual(run(check()), (400, 400))
        refused = [fields for name, fields in self.events() if name == "req_refused"]
        self.assertEqual(len(refused), 2)
        self.assertEqual({f["status"] for f in refused}, {"400"})
        reasons = " ".join(f.get("reason", "") for f in refused)
        self.assertIn("JSONDecodeError(line=1", reasons)     # reduced to line/column, not the body text
        self.assertNotIn("model", reasons)

    def test_image_request_logs_cache_hits_misses_and_budget(self):
        os.environ["KYOJIN_REQLOG"] = str(self.log_path)
        from PIL import Image
        from aiohttp.test_utils import TestClient, TestServer
        eng, _ = vision_engine()

        class VisionEngine(FakeEngine):
            supports_vision = True

            def image_tokens(self, image):
                return 3

            async def generate(self, prompt, images=None, img_stats=None, **kw):
                if images:
                    serve.ResidentEngine._embed(eng, images, img_stats)
                async for piece in FakeEngine.generate(self, prompt, **kw):
                    yield piece

        def uri(color):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(buf, "PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": uri((255, 0, 0))}},
            {"type": "image_url", "image_url": {"url": uri((0, 0, 255))}}]}]}

        async def check():
            client = TestClient(TestServer(serve.create_app(VisionEngine("hello"), "m", IMAGE_TEMPLATE)))
            await client.start_server()
            for _ in range(2):                                     # second send: both pictures are held
                response = await client.post("/v1/chat/completions", json=body)
                await response.read()
            await client.close()

        run(check())
        cache = [fields for name, fields in self.events() if name == "img_cache"]
        self.assertEqual(len(cache), 2)
        self.assertEqual((cache[0]["imgs"], cache[0]["hits"], cache[0]["misses"]), ("2", "0", "2"))
        self.assertEqual(cache[0]["first_missing"], "0")           # the prefix can only match from the start
        self.assertEqual((cache[0]["entries"], cache[0]["evicted"]), ("2", "0"))
        self.assertEqual(cache[0]["budget_mb"], "1024")            # the shared cache's own default (1 GiB)
        self.assertEqual(cache[0]["item_cap"], "32")            # the shared cache's count default
        self.assertEqual((cache[1]["hits"], cache[1]["misses"]), ("2", "0"))
        self.assertNotIn("first_missing", cache[1])                # a full hit has no gap
        self.assertNotIn("data:image", "\n".join(self.lines()))     # the image URL is never written

    def test_heartbeat_while_a_job_runs(self):
        os.environ["KYOJIN_REQLOG"] = str(self.log_path)
        os.environ["KYOJIN_REQLOG_HEARTBEAT_S"] = "0.05"
        from aiohttp.test_utils import TestClient, TestServer

        class Slow(FakeEngine):
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(0.3)
                async for piece in FakeEngine.generate(self, prompt, **kwargs):
                    yield piece

        async def check():
            client = TestClient(TestServer(serve.create_app(Slow("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}]})
            await response.read()
            await client.close()

        run(check())
        alive = [fields for name, fields in self.events() if name == "alive"]
        self.assertTrue(alive, "no heartbeat lines")
        self.assertTrue(all("elapsed_s" in f and "phase" in f for f in alive))
        self.assertTrue({f["phase"] for f in alive} <= {"parse", "queued", "prefill", "decode", "stream"})

    def test_v1_completions_is_logged_too(self):
        os.environ["KYOJIN_REQLOG"] = str(self.log_path)
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("hello"), "m", TEMPLATE)))
            await client.start_server()
            response = await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 4})
            await response.read()
            await client.close()
            return response.status

        self.assertEqual(run(check()), 200)
        queued = [fields for name, fields in self.events() if name == "req_queued"]
        self.assertEqual([f["path"] for f in queued], ["/v1/completions"])
        self.assertIn("job_done", self.names())


if __name__ == "__main__":
    unittest.main()
