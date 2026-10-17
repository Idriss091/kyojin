# ds41-serve — DeepSeek-V4.1-Flash-1017 OpenAI-compatible server

`tools/ds41/serve.py` is an aiohttp server in the shape of [tools/glm/serve.py](../glm/SERVE.md): the same
queue, `cache_salt`, `timings` / `usage`, request log, SSE keep-alive and start-up `/health`. Text in, text and
tool calls out (no images). The model is behind a small engine contract (`ds41_engine.py`), so the HTTP layer,
the prompt renderer and the output parser run and are tested without a GPU.

| File | Role |
|---|---|
| `serve.py` | HTTP layer, flags, sampling defaults |
| `ds41_chat.py` | prompt renderer: port of the official `encoding/encoding_dsv4.py` plus the OpenAI mapping |
| `ds41_parse.py` | reply parser: thinking / answer split, DSML tool calls, streaming |
| `ds41_engine.py` | engine contract, `load_model` hook, `StubEngine` |
| `test_ds41_*.py` | pytest suites (CPU, no weights) |
| `fetch_reference.py` | downloads the small reference files (no weights) |

## Start

```bash
python3 tools/ds41/serve.py --model <model-folder> --loader <module:function> --port 8000 -c 131072
# without a GPU, canned replies:
python3 tools/ds41/serve.py --model <model-folder> --loader stub --engine-opt replies="Hello from the stub"
```

`<model-folder>` needs `config.json` and `tokenizer.json` (and `generation_config.json` when the pack has one).

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `~/models/deepseek-v4.1-flash-1017` | model folder |
| `--model-id` | `deepseek-v4.1-flash` | id in `/v1/models` and in replies; a request naming another model gets 400 |
| `--host`, `--port` | `127.0.0.1`, `8000` | listen address |
| `-c`, `--max-ctx` | `131072` | context window in tokens |
| `--loader` | `ds41_model:load_model` | `module:function` that builds the engine; `stub` = canned replies |
| `--engine-opt KEY=VALUE` | none | extra keyword argument for the loader (JSON value or plain string), repeatable |
| `--default-temperature` | pack file, else `1.0` | used when a request omits `temperature` |
| `--default-top-p` | pack file, else `1.0` | used when a request omits `top_p` |
| `--default-max-tokens` | all free context | reply budget (thinking included) when a request omits it |
| `--default-reasoning-effort` | `high` | `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or an integer 1..100 |
| `--default-thinking` | `on` | `on` / `off`, when a request does not choose |

Request log, SSE keep-alive and start-up progress are the shared code described in
[doc/logging.md](../../doc/logging.md) and [doc/env_vars.md](../../doc/env_vars.md).

## Sampling defaults

For `temperature` and `top_p`, in order: the request, a flag, the `generation_config.json` of the model folder,
the model card values (`temperature 1.0`, `top_p 1.0`). The Hugging Face repo of the model ships no
`generation_config.json`, so without one in the pack the model card values apply. A request value of `null`
counts as omitted.

## Endpoints

* `GET /health`, `GET /v1/models`, `GET /metrics` — as on the GLM server.
* `POST /v1/chat/completions` — streaming and not. `reasoning_content`, `tool_calls`, `usage`, `timings`
  (`cache_n`, `prompt_n`, ... as llama.cpp), `cache_prompt`, `cache_salt`, `stop`, `max_tokens` /
  `max_completion_tokens` (must be integers >= 1), `n` must be 1.
* `POST /v1/completions` — verbatim prompt (text or token ids), no template, no BOS added.
* `POST /completion` — minimal llama.cpp form of the same.
* `POST /tokenize` — `{prompt}` or `{messages, tools, ...}`. With `messages` the chat prompt is rendered, BOS
  included, and `count` equals the `prompt_tokens` of the same chat request.
* `POST /apply-template` — returns the rendered prompt, generates nothing.

Finish reasons: `stop`, `length` (budget used, or a tool call cut by the budget), `tool_calls`.

## Prompt format

`ds41_chat.render_chat` ports the official encoder and is checked against it, string for string and token id for
token id (`test_ds41_chat.py`). Turn markers are `<｜User｜>`, `<｜Assistant｜>`, `<｜System｜>`; the generation
prompt ends in `<think>` (thinking) or `</think>` (chat). One BOS opens the prompt.

Handled like the reference: system, user, assistant (with `reasoning_content`), multi-turn, tools (system block
`## Tools`), tool calls (DSML), tool results (grouped into the next user turn, in call order), thinking
dropped from earlier turns (`drop_thinking`, switched off automatically when tools are present).

OpenAI layer on top:

* `developer` role is `system`. `content: null` is empty. Text-part lists are joined. Image parts get 400.
* Tools given but no system message: an empty system message is inserted to carry them.
* `tools` accepts the nested (`function: {...}`) and the flat form. `tool_choice: "none"` removes the tools from
  the prompt and turns off call parsing.
* Assistant `tool_calls[].function.arguments` may be a JSON string or an object. A `namespace` field, or a
  `ns::name` function name, renders the `ns::name` form.
* A prompt that ends on an assistant turn, or has only a system message, gets 400.

### Thinking and effort mapping

Where each setting is read, first hit wins: `chat_template_kwargs`, then the top-level body field, then the
server default.

| Request | Prompt |
|---|---|
| nothing | thinking, effort 75 (`--default-*` change this) |
| `reasoning_effort: "none"` | chat mode (no thinking), no effort line |
| `reasoning_effort: "minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` / `"max"` | thinking with effort 25 / 50 / 62 / 75 / 100 / 100 |
| `reasoning_effort: 1..100` | thinking with that effort |
| `thinking: false`, `{"type": "disabled"}`, `enable_thinking: false`, `thinking_mode: "chat"` | chat mode |
| `thinking: true`, `{"type": "enabled"}`, `enable_thinking: true`, `thinking_mode: "thinking"` | thinking |
| `drop_thinking: false` | keep earlier reasoning blocks in the prompt |

An explicit thinking switch beats `reasoning_effort: "none"`. `low`, `high` and `max` are the official mapping
(50, 75, 100); `minimal`, `medium` and `xhigh` are this server's choice. In thinking mode the effort line
(`Reasoning Effort: N (range 1-100, ...)`) goes at the top of the first system message. Bad values get 400.

## Reply parsing

`ds41_parse.StreamParser` reads the decoded text, text-only and incremental:

* thinking mode: text up to `</think>` is `reasoning_content`, the rest is `content`; chat mode: all `content`;
* a `<｜DSML｜ calls>` block becomes `tool_calls` (`id`, `function.name`, JSON `arguments`, `namespace` when the
  name is `ns::name`);
* streaming sends `tool_calls` deltas: the first has `index`, `id`, `type`, `name` and empty `arguments`, then
  `arguments` pieces that add up, byte for byte, to the official parser's JSON; markup is never sent as content;
* only a possible start of a marker is held back, so text streams character by character;
* a reply cut inside a call closes the JSON, and the finish reason is `length`;
* malformed markup falls back to plain content; the parser does not raise.

## Engine contract

`ds41_engine.Engine` lists what `serve.py` calls: `ctx`, `sessions`, `vocab_size`, `count_tokens`, `prompt_ids`,
`encode_ids`, `generate`, and optionally `reset_prefix_cache`. Rules for a real engine:

* `generate` yields decoded text **with special tokens kept** (`</think>`, `｜DSML｜`). A decoder that skips
  special tokens breaks reasoning and tool calls.
* `prompt_ids` tokenizes a rendered prompt with special tokens parsed and no extra BOS (the prompt has one).
* `stats` / `last_stats` get `cached_tokens`, `new_tokens`, `time_prefill`, `time_generate` at the end.
* `reset_prefix_cache` is called when `cache_salt` changes.

`StubEngine` implements it over canned replies (`--loader stub`, or `StubEngine(replies)` in code).

## Tests

```bash
python3 -m pytest tools/ds41 -q
```

CPU only, no weights. Golden tests compare against the vendored official encoder
(`testdata/reference/encoding.py`, MIT, with the official fixtures). Token-id equality needs the model's
`tokenizer.json`: set `DS41_REF_DIR` to a folder holding it, or run `python3 tools/ds41/fetch_reference.py <dir>`
(downloads only config, tokenizer and encoding files, never weights). Without it those tests are skipped and
reported as skipped.
