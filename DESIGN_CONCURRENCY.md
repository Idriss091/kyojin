# Concurrent requests in the GLM and Qwen servers

Status: the batched decode path exists (`--sessions N`). This branch adds the `--max-batch N` spelling, CPU tests for the
scheduler, and this note. The default (1) takes the same code path as before.

## 1. What the servers do today

Both servers (`tools/glm/serve.py`, `tools/qwen/serve.py`) have two paths, chosen by `--sessions`:

| | `--sessions 1` (default) | `--sessions N > 1` |
|---|---|---|
| Admission | one worker, one `asyncio.Lock`: requests run one at a time, in order | N workers, one `Semaphore(N)`, `admit_lock` orders the start |
| Generator use | the request thread calls `generator.iterate()` itself | one driver task (`_drive_loop`) owns the Generator. Requests post commands (`enqueue`, `cancel`, image tower, slot actions) and read their own events from a queue, routed by job serial number |
| Batching | none | the Generator's own batch: all active jobs advance in one `iterate()` call, up to `Cache(max_batch_size=N)` slots |
| Cancel | `generator.cancel(job)` between steps | `forget()` on the driver between steps; the slot is free for the next queued job |
| Speculative decode | MTP drafter (`--num-draft`, default 2) and n-gram lookup (`GLM_NGRAM`, default `2,7,16,3`) | MTP kept, depth capped by `--row-budget` (default 5), n-gram lookup off |

Per request state inside the Generator (`exllamav3/generator/generator.py`, `job.py`):

- **Page table.** The KV pool is one paged pool of `ctx + 4096` tokens shared by all jobs. Each job owns a list of pages.
  Pages are keyed by content hash, so a finished prompt stays reusable (prompt cache). With N busy jobs the per-job
  window is `ctx / N`.
- **Recurrent state.** GLM (KDA layers) and Qwen (linear attention) keep a fixed-size state per job: one slot per
  session and one copy per history level (138 MiB each in GLM). The history levels are the rollback points a rejected
  draft needs. Checkpoints at prompt end make the prompt cache work for these layers.
- **MTP drafter.** It has its own Cache with the same slot layout. Each job drafts for itself; verify runs the target on
  `1 + depth` rows per job in one forward pass.
- **Sparse attention indexer (DSA).** Per row: each row picks its own top-k tokens, so its cost grows with the row count,
  not with the weight bytes.

The external benchmark (C1 24.9 / C2 25.7 / C4 25.1 tok/s) ran the default `--sessions 1`. With `--sessions 4` the repo
already measures 30 / 37.5 / 40.6 tok/s total for 1 / 2 / 4 requests (`doc/benchmarks.md`).

## 2. What limits the gain

1. **Speculative verify is per stream.** A round verifies `1 + depth` rows for each job, and each job accepts a
   different number of tokens, so the streams do not stay in step. The Generator handles this (each job rolls back
   its own recurrent history level), but the rows add up: `R = B x (depth + 1)`.
2. **MoE experts do not share.** The weights are read once per step, but only the dense part (attention, shared
   expert, router, lm head) is shared by all rows. The routed experts are read per distinct expert: more rows touch more
   experts. This is why batching a MoE model gains much less than batching a dense one.
3. **Recurrent state and drafter catch-up.** Memory grows with `slots x history levels`. n-gram lookup rounds need a
   single active request (the lookup tree is one request's), so they are off at batch > 1. This costs +26 % on copy and
   edit turns.
4. **Attention kernel split.** The decode attention kernel picks its split layout from the batch, so a stream's fp16
   result can differ from a solo run at an exact tie. The servers print a warning at start. Greedy equality is not
   guaranteed on GPU. The CPU tests here prove scheduling, routing and cancellation, not these numerics.
5. **Whole-engine actions.** Slot save/restore waits for every running request (they take all N permits). A `cache_salt`
   change rebuilds the Generator and drops the prompt cache and, if jobs are in flight, the other streams. The Qwen server
   rejects `speculative=false` and `cache_prompt=false` at N > 1 with 400 for the same reason.
6. **Shared KV pool.** N long prompts need `N x prompt` pages; a request that does not fit waits.

## 3. Bandwidth model

Decode is bandwidth bound. One step reads the dense part `D` once and the distinct routed experts `X x u(R)`:

    bytes(R)  = D + X * u(R)
    u(R)      = 1 - (1 - k/E)^R          (R independent rows, E experts, k per row)
    tok/s     = BW / bytes(R) * B * tokens_per_stream_step
    tokens_per_stream_step = 1 + a + a^2 + ... + a^depth      (a = draft acceptance)

Inputs: pack 100 GB, BW 230 GB/s, E = 256 and k = 8 (assumed GLM-class routing; the sparsity of the real model may
differ), a = 0.79 on chat (SERVE.md: 0.79 chat, 0.57 prose). Calibrating on the measured single-stream 30 tok/s at depth 2
(R = 3, 2.4 tokens per step, 12.5 steps/s, 18 GB per step) gives D = 9.8 GB and X = 90 GB. Independent rows are an upper
bound for the union (adjacent tokens share experts), so real unions are smaller and the dense part larger.

Model result (tok/s total, a = 0.79 / a = 0.57):

| B | depth | rows R | GB/step | a = 0.79 | a = 0.57 | measured (chat) |
|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 18.0 | 30.9 | 24.2 | 30 |
| 1 | 0 | 1 | 12.6 | 18.2 | 18.2 | |
| 2 | 1 | 4 | 20.5 | 40.1 | 35.2 | 37.5 |
| 2 | 0 | 2 | 15.3 | 30.0 | 30.0 | |
| 4 | 1 | 8 | 30.0 | 54.9 | 48.2 | 40.6 |
| 4 | 0 | 4 | 20.5 | 44.8 | 44.8 | |
| 4 | 2 | 12 | 38.3 | 58.0 | 45.5 | |

Reading it:

- Batching 2 streams fits the measurement (40 vs 37.5). At B = 4 the model says 55 and the box measures 40.6. The gap
  (about 26 %) is cost that is not weight bandwidth and grows with rows: the DSA indexer, per-row attention over the
  KV, expert-gather and launch overhead. So the practical ceiling is about 1.35x at B = 4, not 4x.
- The model ranks speculative decode above plain decode at every B (4 x depth 1 beats 4 x depth 0: 55 vs 45) because a
  draft row costs far less than a new stream (the expert union grows slowly). That ranking holds while the step is
  bandwidth bound. If the non-bandwidth cost is per row, then at B = 4 the 8-row step costs about twice the 4-row step in
  overhead, and the plain 4-row step could win. The measured 40.6 against a model that says 44.8 for plain makes the two
  too close to call. This needs a GPU A/B at B = 4: `--row-budget 4` (depth 0 is not reachable today, minimum depth is 1) against
  depth 1, same prompts, 3 runs each.
- At B = 2 keep depth 1. At B >= 3 `--row-budget 5` already sets depth 1 (for B = 4: `5 // 4 - 1 = 0`, floored to 1).

Decision: keep speculative decode per stream (depth lowered by `--row-budget`), as today. Do not disable it at batch > 1
until the GPU A/B above says otherwise. Switching the drafter off in the middle of a run is not free: the drafter Cache
must still see every accepted token (catch-up path, `mtp_defer`), so a plain step with a loaded drafter is a code change
in the Generator, not a server flag. A server restart with `--num-draft 0` gives the plain-decode number for the A/B.

## 4. Minimal design for continuous batching of up to 4 streams

Already in place (reused, unchanged):

1. **Admission.** The app accepts requests into a FIFO queue; N workers each run one request inside `generate()`.
   The Generator keeps jobs pending until a slot and enough pages are free (`iterate_start_jobs`), in queue order.
2. **Per-sequence state.** One recurrent slot (with history levels) and one drafter slot per job; page table per job;
   prompt cache by content hash shared by all jobs.
3. **Batched step.** One driver task calls `iterate()`; all active jobs advance together (prefill chunks and decode
   rows are mixed by the Generator).
4. **Streaming.** Events are routed by serial number to a per-request queue; the HTTP handler reads only its own.
5. **Cancellation.** A request that ends for any reason (client gone, error, EOS) posts `forget()`; the driver cancels
   the job between two steps. A request cancelled while still queued never starts.
6. **Draft depth.** `--row-budget` lowers the depth when more streams are busy so verify rows stay bounded.

Added on this branch:

- `--max-batch N` in both servers, same meaning and range as `--sessions`; giving both with different values is an
  error. Default 1 stays the old single-request path: no line of that path changed.
- `tools/glm/test_batching.py`: the scheduler tests below.

Not done, needs GPU work:

- A/B of plain vs depth-1 decode at B = 4 (section 3).
- n-gram lookup per stream (the lookup tree is per request; the round shape is not uniform across streams).
- A batch-invariant attention split so greedy output at N > 1 equals the solo run bit for bit.
- Slot save/restore of one stream without draining the others.

## 5. Tests

`tools/glm/test_batching.py` (CPU, no model) runs the real `_run_shared` and `_drive_loop` against a fake Generator
with the same contract (FIFO pending queue, slot cap, one batched step per `iterate()`, speculative rounds of varying
length, prefix cache). The fake model is a pure function of the context, so equality with a sequential run holds by
construction on the model side and fails if routing, admission or cancellation mixes streams up.

- concurrent output equals sequential output for 6 requests at 1, 2 and 4 slots (greedy);
- decode steps are shared: 4 streams need fewer than half the steps of 1-slot serving;
- FIFO admission order and no starvation with more requests than slots;
- cancel mid-stream: the other streams finish with correct tokens, the slot frees, routes empty;
- cancel while queued: the request never starts;
- prompt-cache reuse across concurrent streams reports the shared prefix;
- `--row-budget` lowers draft depth without changing tokens;
- `--max-batch` / `--sessions` resolution.

Run: `python -m pytest tools/glm/test_batching.py`.
