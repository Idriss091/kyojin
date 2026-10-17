#!/usr/bin/env python3
"""Continuous batching (--max-batch / --sessions) with a model-free fake Generator. CPU only.

The fake Generator follows the contract the engine driver relies on: enqueue() returns a serial number, jobs wait in
FIFO order until a slot is free, every iterate() advances all active jobs together (one batched step), a job emits
1 + accepted tokens per step (speculative rounds), cancel() frees the slot, a prefix cache keeps the prompts of
finished jobs. Its "model" is a pure function of the token context, so a stream's tokens cannot depend on its
neighbours: equality with a sequential run proves the scheduling and the routing, not the numerics of the GPU
kernels at batch > 1.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve

VOCAB = 1000


def next_token(ctx: list[int]) -> int:
    """Greedy 'model': depends on the last three tokens only."""
    a = ctx[-3:]
    return (sum((i + 7) * t for i, t in enumerate(a)) * 31 + len(a)) % VOCAB


class Ids:
    def __init__(self, ids):
        self.ids = list(ids)

    def numel(self):
        return len(self.ids)


class FakeJob:
    serial_number = None

    def __init__(self, ids: list[int], max_new_tokens: int):
        self.ids, self.max_new_tokens = list(ids), max_new_tokens
        self.out: list[int] = []
        self.cached = 0


class FakeGenerator:
    def __init__(self, slots: int):
        self.slots = slots
        self.pending_jobs: list[FakeJob] = []
        self.active_jobs: list[FakeJob] = []
        self.deferred_jobs = None
        self.draft_model = None
        self.num_draft_tokens = 2
        self.serial = 0
        self.prompts: list[list[int]] = []     # prefix cache: prompts of finished jobs
        self.step_sizes: list[int] = []        # active jobs per iterate()
        self.start_order: list[int] = []
        self.depths: list[int] = []

    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)

    def enqueue(self, job):
        self.serial += 1
        job.serial_number = self.serial
        self.pending_jobs.append(job)
        return job.serial_number

    def cancel(self, job):
        for lst in (self.pending_jobs, self.active_jobs):
            if job in lst:
                lst.remove(job)

    def clear_queue(self):
        self.pending_jobs.clear()
        self.active_jobs.clear()

    def _reuse(self, ids):
        best = 0
        for p in self.prompts:
            n = 0
            while n < min(len(p), len(ids) - 1) and p[n] == ids[n]:
                n += 1
            best = max(best, n)
        return best

    def iterate(self):
        while self.pending_jobs and len(self.active_jobs) < self.slots:   # FIFO admission
            j = self.pending_jobs.pop(0)
            j.cached = self._reuse(j.ids)
            self.active_jobs.append(j)
            self.start_order.append(j.serial_number)
        self.step_sizes.append(len(self.active_jobs))
        self.depths.append(self.num_draft_tokens)
        events = []
        for j in list(self.active_jobs):
            accepted = min(self.num_draft_tokens, (len(j.out) + j.serial_number) % 3)   # varies with depth, never the tokens
            text = ""
            for _ in range(1 + accepted):
                if len(j.out) >= j.max_new_tokens:
                    break
                t = next_token(j.ids + j.out)
                j.out.append(t)
                text += f"{t} "
            ev = {"serial": j.serial_number, "text": text}
            if len(j.out) >= j.max_new_tokens:
                ev.update(eos=True, new_tokens=len(j.out), cached_tokens=j.cached)
                self.active_jobs.remove(j)
                self.prompts.append(list(j.ids))
            events.append(ev)
        return events


def make_engine(slots: int, row_budget: int = 0, drafter: bool = False):
    eng = object.__new__(serve.ResidentEngine)
    eng.sessions = slots
    eng.row_budget = row_budget
    eng.max_draft = 2
    eng.greedy_generator = FakeGenerator(slots)
    if drafter:
        eng.greedy_generator.draft_model = object()
    return eng


async def stream(eng, ids, n, cancel=None, after=None):
    """One request: returns (tokens, stats). after = set `cancel` once that many pieces have arrived."""
    st, toks, pieces = {}, [], 0
    async for piece in eng._run_shared(FakeJob(ids, n), Ids(ids), cancel, st, 0.0, None):
        toks += [int(x) for x in piece.split()]
        pieces += 1
        if after is not None and pieces >= after:
            cancel.set()
    return toks, st


def solo(ids, n):
    out = []
    while len(out) < n:
        out.append(next_token(ids + out))
    return out


PROMPTS = [[1, 2, 3, 4], [9, 8, 7], [5, 5, 5, 5, 5, 5], [42], [100, 200, 300, 400, 500], [7, 7]]


class BatchingTests(unittest.TestCase):
    def test_concurrent_equals_sequential(self):
        async def go(slots):
            eng = make_engine(slots)
            res = await asyncio.gather(*(stream(eng, p, 11 + i) for i, p in enumerate(PROMPTS)))
            return [r[0] for r in res], eng
        for slots in (1, 2, 4):
            toks, eng = asyncio.run(go(slots))
            self.assertEqual(toks, [solo(p, 11 + i) for i, p in enumerate(PROMPTS)], f"slots={slots}")
            self.assertEqual(eng.greedy_generator.num_remaining_jobs(), 0)
            self.assertEqual(eng._routes, {})

    def test_decode_steps_are_shared(self):
        async def go(slots):
            eng = make_engine(slots)
            await asyncio.gather(*(stream(eng, p, 24) for p in PROMPTS[:4]))
            return eng.greedy_generator
        one, four = asyncio.run(go(1)), asyncio.run(go(4))
        self.assertEqual(max(one.step_sizes), 1)
        self.assertEqual(max(four.step_sizes), 4)
        self.assertLess(len(four.step_sizes) * 2, len(one.step_sizes))   # weights are read once per step for all streams

    def test_fifo_admission_and_no_starvation(self):
        async def go():
            eng = make_engine(2)
            done = []

            async def one(i, n):
                r = await stream(eng, PROMPTS[i], n)
                done.append(i)
                return r
            tasks = []
            for i, n in enumerate([40, 40, 3, 3, 3, 3]):
                tasks.append(asyncio.create_task(one(i, n)))
                await asyncio.sleep(0)          # submission order = index order
            await asyncio.gather(*tasks)
            return eng, done
        eng, done = asyncio.run(go())
        self.assertEqual(eng.greedy_generator.start_order, [1, 2, 3, 4, 5, 6])   # serials = submission order
        self.assertEqual(sorted(done), list(range(6)))                           # nobody starves

    def test_cancel_mid_stream_keeps_the_others_and_frees_the_slot(self):
        async def go():
            eng = make_engine(2)
            stop = asyncio.Event()
            long_ = asyncio.create_task(stream(eng, PROMPTS[0], 500, cancel=stop, after=3))
            other = asyncio.create_task(stream(eng, PROMPTS[1], 20))
            queued = asyncio.create_task(stream(eng, PROMPTS[2], 15))     # waits for a slot, runs after the cancel
            return eng, await asyncio.gather(long_, other, queued)
        eng, (c, o, q) = asyncio.run(go())
        self.assertLess(len(c[0]), 500)
        self.assertEqual(c[0], solo(PROMPTS[0], 500)[:len(c[0])])         # what it did emit is correct
        self.assertEqual(o[0], solo(PROMPTS[1], 20))
        self.assertEqual(q[0], solo(PROMPTS[2], 15))
        self.assertEqual(eng.greedy_generator.num_remaining_jobs(), 0)
        self.assertEqual(eng._routes, {})

    def test_cancel_while_queued_never_starts(self):
        async def go():
            eng = make_engine(1)
            stop = asyncio.Event()
            stop.set()                     # the client is gone before its request leaves the queue
            a = asyncio.create_task(stream(eng, PROMPTS[0], 200))
            b = asyncio.create_task(stream(eng, PROMPTS[1], 12, cancel=stop))
            return eng, await a, await b
        eng, a, b = asyncio.run(go())
        self.assertEqual(a[0], solo(PROMPTS[0], 200))
        self.assertEqual(b[0], [])
        self.assertEqual(eng.greedy_generator.start_order, [1])
        self.assertEqual(eng.greedy_generator.num_remaining_jobs(), 0)

    def test_prompt_cache_reuse_across_concurrent_streams(self):
        base = list(range(10, 40))

        async def go():
            eng = make_engine(4)
            first = await stream(eng, base + [1], 6)
            a, b = await asyncio.gather(stream(eng, base + [2], 6), stream(eng, base + [3], 6))
            return first, a, b
        first, a, b = asyncio.run(go())
        self.assertEqual(first[1]["cached_tokens"], 0)
        self.assertEqual(a[1]["cached_tokens"], 30)
        self.assertEqual(b[1]["cached_tokens"], 30)
        self.assertEqual(a[0], solo(base + [2], 6))
        self.assertEqual(b[0], solo(base + [3], 6))

    def test_row_budget_lowers_draft_depth_but_not_tokens(self):
        async def go(budget):
            eng = make_engine(4, row_budget=budget, drafter=True)
            res = await asyncio.gather(*(stream(eng, p, 16) for p in PROMPTS[:4]))
            return [r[0] for r in res], eng.greedy_generator
        capped, gen = asyncio.run(go(5))
        free, gen0 = asyncio.run(go(0))
        self.assertEqual(capped, free)
        self.assertEqual(capped, [solo(p, 16) for p in PROMPTS[:4]])
        self.assertEqual(gen.depths[0], 1)         # 4 busy streams x (1 + 1) rows
        self.assertEqual(gen0.depths[0], 2)

    def test_max_batch_flag_is_sessions(self):
        def parse(*argv):
            p = argparse.ArgumentParser()
            p.add_argument("--sessions", type=int, default=1)
            p.add_argument("--max-batch", type=int, default=None)
            a = p.parse_args(argv)
            serve.resolve_batch_flags(p, a)
            return a.sessions
        self.assertEqual(parse(), 1)
        self.assertEqual(parse("--max-batch", "4"), 4)
        self.assertEqual(parse("--sessions", "3"), 3)
        self.assertEqual(parse("--sessions", "4", "--max-batch", "4"), 4)
        with self.assertRaises(SystemExit):
            parse("--sessions", "2", "--max-batch", "4")


if __name__ == "__main__":
    unittest.main()
