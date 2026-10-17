"""DSpark speculative decoding controller for DeepSeek-V4.1-Flash-1017 (framework agnostic).

What the official code gives us
-------------------------------
The official `inference/` folder ships the DSpark *forward pass* only (`Transformer.forward_spec`, `DSparkBlock`).
Its README says generation is plain autoregressive and `model.py` says nothing calls `forward_spec`. So there is no
official draft/verify loop, no official acceptance rule and no official cache rollback. This module is that loop,
built on the contract `forward_spec` defines:

* The draft stack (`n_mtp_layers` = 3 `DSparkBlock` layers, run as one network) takes ONE block per call: the token the
  target just chose (the "anchor", at position L) followed by `dspark_block_size - 1` noise tokens
  (`dspark_noise_token_id`). All positions of the block attend to each other and to a sliding window of main-model
  context, projected from `main_hidden`: the attention inputs (mean over the hyper-connection copies) of the target
  layers `dspark_target_layer_ids`, concatenated, at position L-1.
* The output is `block_size` draft tokens for positions L+1 .. L+block_size. Each one is chosen from the stage logits
  plus a rank-`dspark_markov_rank` bigram bias that depends on the previous drafted token (the Markov head), so the
  block is chained. A confidence head also scores each position; the official code does not say how to use it, and
  this module does not.
* The stack keeps one window-KV slot per context position, written from `main_hidden` of that position. Each call
  writes exactly one slot, so the drafter must see the hidden state of EVERY accepted position, in order. A skipped
  position leaves a hole in its window and lowers acceptance (it never makes the output wrong).

The loop (greedy) per step, with `tok` at position L not yet in the target cache:

1. `drafts = drafter.draft(tok, hidden_rows, L - 1)` -> up to `block_size` tokens, d1..dK for positions L+1..L+K.
2. `out = target.verify([tok, d1..dK], L)` -> logits and target-layer hidden states for those K+1 positions.
3. Accept d_k while `argmax(out.logits[k-1]) == d_k`. The token after the last accepted one is `argmax` of the next
   logits row (a correction on a mismatch, a free bonus token when all K matched). A step emits a+1 tokens.
4. `target.truncate(L + a + 1)` drops the rejected positions; the new token becomes `tok`; `hidden_rows` are the rows
   of the a+1 kept positions. The drafter needs no rollback: it is only ever fed accepted positions.

Output is token identical to plain greedy decoding, for any drafter.

Sampling. The official sampler (`model.sample`) is the Gumbel-max trick. With a deterministic (argmax) draft the exact
speculative sampling rule is: draw the target token for each position in order with the official rule, accept the draft
if it equals the draw, stop at the first mismatch. That is standard speculative sampling for a one-hot draft law
(accept with probability p(d), residual = p without d), and it consumes one noise draw per emitted token, so under the
same seed the output matches the plain `generate()` loop token for token. When the drafter returns the laws `q` it drew
from (`DraftOut.q`), the standard rule is used instead: accept with probability min(1, p/q), resample from the
normalised max(0, p - q) on rejection; same output distribution, no seed parity with the plain loop.

Plain EOS handling follows `generate.py`: a token counts against `max_tokens` when it is sampled; the EOS token is
not part of the returned completion; the last emitted token is never in the target cache.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["TargetOut", "DraftOut", "Target", "Drafter", "Sampler", "GreedySampler", "GumbelSampler",
           "DSparkStats", "DSparkDecoder", "decode"]


# ---- interfaces ---------------------------------------------------------------------------------------------------


@dataclass
class TargetOut:
    """`logits[i]` is the next-token logit row after input position i; `hidden[i]` is the target-layer hidden state
    (what `Transformer.forward` returns as `main_hidden`) of input position i. Any 2-D indexable and sliceable
    object works (torch tensor, numpy array, list of rows)."""
    logits: Any
    hidden: Any


@dataclass
class DraftOut:
    """Draft tokens for positions L+1.. . `q[i]`, when given, is the probability vector `tokens[i]` was drawn from."""
    tokens: list[int]
    q: Sequence[Any] | None = None


@runtime_checkable
class Target(Protocol):
    """The big model, with a KV cache that holds the committed positions 0..n-1."""

    def prefill(self, tokens: Sequence[int]) -> TargetOut:
        """Reset the cache and run `tokens` at positions 0..P-1. `logits` needs only the last row (the official prefill
        returns just that one); `hidden` has P rows."""
        ...

    def verify(self, tokens: Sequence[int], start_pos: int) -> TargetOut:
        """Run `tokens` at positions start_pos.. on top of a cache of length start_pos. One logits row and one hidden
        row per token. The cache then holds start_pos + len(tokens) positions."""
        ...

    def truncate(self, length: int) -> None:
        """Make the cache equal to a cache that only ever saw positions 0..length-1. Called with a length that is not
        below the start of the last `verify` and not above its end."""
        ...


@runtime_checkable
class Drafter(Protocol):
    """The DSpark stack."""

    block_size: int       # `dspark_block_size`: the most tokens one call proposes

    def draft(self, token: int, hidden: Any, start_pos: int) -> DraftOut:
        """`token` is the anchor, at position start_pos + 1. `hidden` holds the target-layer hidden rows of the
        positions the drafter has not seen yet, ending at start_pos: rows for positions start_pos-n+1 .. start_pos.
        The first call gets the whole prompt (prefill seeds the window); later calls get the kept positions of the
        last step. Returns up to `block_size` tokens for positions start_pos + 2 .."""
        ...


@runtime_checkable
class Sampler(Protocol):
    stochastic: bool

    def sample(self, logits_row: Any) -> int:
        """One token from a logits row. Consumes one random draw when stochastic."""
        ...

    # only used for drafts that come with their laws `q`
    def probs(self, logits_row: Any) -> Any: ...
    def uniform(self) -> float: ...
    def draw(self, probs: Any) -> int: ...


class GreedySampler:
    stochastic = False

    def sample(self, logits_row: Any) -> int:
        return int(logits_row.argmax())

    def probs(self, logits_row: Any) -> Any:
        raise NotImplementedError("greedy decoding has no probabilities")

    uniform = draw = probs


class GumbelSampler:
    """The official `model.sample` rule: Gumbel-max over softmax(logits / temperature), one exponential draw per call.
    Uses torch's global RNG unless a `torch.Generator` is given. temperature 0 is argmax."""
    stochastic = True

    def __init__(self, temperature: float = 1.0, generator: Any = None):
        self.temperature = temperature
        self.generator = generator

    def sample(self, logits_row: Any) -> int:
        import torch
        if self.temperature == 0:
            return int(logits_row.argmax())
        logits = logits_row.reshape(1, -1) / max(self.temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        noise = torch.empty_like(probs).exponential_(1, generator=self.generator)
        return int(probs.div_(noise).argmax(dim=-1))

    def probs(self, logits_row: Any) -> Any:
        import torch
        return torch.softmax(logits_row.reshape(-1) / max(self.temperature, 1e-5), dim=-1, dtype=torch.float32)

    def uniform(self) -> float:
        import torch
        return float(torch.rand(1, generator=self.generator))

    def draw(self, probs: Any) -> int:
        import torch
        return int(torch.multinomial(probs, 1, generator=self.generator))


# ---- statistics ---------------------------------------------------------------------------------------------------


@dataclass
class DSparkStats:
    steps: int = 0                  # verify rounds that had at least one draft token
    plain_steps: int = 0            # rounds without a draft (the last token of a budget)
    proposed: int = 0               # draft tokens sent to verify
    accepted: int = 0               # draft tokens equal to the target's choice
    emitted: int = 0                # tokens returned, EOS included
    draft_len_hist: Counter = field(default_factory=Counter)      # draft length -> rounds
    accepted_hist: Counter = field(default_factory=Counter)       # accepted drafts -> rounds

    @property
    def rejected(self) -> int:
        return self.proposed - self.accepted

    @property
    def accepted_per_step(self) -> float:
        """Mean accepted draft tokens per drafting round (the bonus/correction token not counted)."""
        return self.accepted / self.steps if self.steps else 0.0

    @property
    def tokens_per_step(self) -> float:
        """Mean tokens emitted per drafting round (accepted + the correction or bonus token)."""
        return (self.accepted + self.steps) / self.steps if self.steps else 0.0

    def as_engine_stats(self) -> dict[str, Any]:
        """The keys tools/glm/serve.py reads from an engine job (`accepted_draft_tokens`, `rejected_draft_tokens`)
        plus the step count and the per-step mean."""
        return {"accepted_draft_tokens": self.accepted, "rejected_draft_tokens": self.rejected,
                "draft_steps": self.steps, "accepted_per_step": round(self.accepted_per_step, 4),
                "draft_len_hist": dict(sorted(self.draft_len_hist.items()))}


# ---- the loop -----------------------------------------------------------------------------------------------------


class DSparkDecoder:
    """Runs one request. `num_draft` caps the draft length (default and maximum: `drafter.block_size`)."""

    def __init__(self, target: Target, drafter: Drafter, *, eos_id: int, sampler: Sampler | None = None,
                 num_draft: int | None = None):
        block = int(drafter.block_size)
        if num_draft is not None and num_draft < 1:
            raise ValueError("num_draft must be >= 1 (use plain decoding for 0)")
        self.target, self.drafter, self.eos_id = target, drafter, eos_id
        self.sampler = sampler or GreedySampler()
        self.num_draft = min(num_draft or block, block)
        self.stats = DSparkStats()

    # -- one verify round --------------------------------------------------------------------------------------------

    def _walk(self, out: TargetOut, draft: DraftOut, k: int, budget: int) -> tuple[list[int], int]:
        """Pick the tokens of one round from the K+1 logits rows. Returns (tokens, matches): `tokens` stops at the first
        mismatch, at EOS, or when `budget` tokens are out; `matches` counts drafts equal to the chosen token."""
        s = self.sampler
        use_q = s.stochastic and draft.q is not None
        tokens: list[int] = []
        matches = 0
        for i in range(k + 1):
            row = out.logits[i]
            d = draft.tokens[i] if i < k else None
            if use_q and d is not None:
                p, q = s.probs(row), draft.q[i]
                pd, qd = float(p[d]), float(q[d])
                if qd > 0 and s.uniform() * qd <= pd:
                    tok, hit = d, True
                else:
                    resid = (p - q).clamp(min=0) if hasattr(p, "clamp") else (p - q).clip(min=0)
                    tot = float(resid.sum())
                    tok, hit = (s.draw(resid / tot) if tot > 0 else s.draw(p)), False
            else:
                tok = s.sample(row)
                hit = d is not None and tok == d
            tokens.append(tok)
            matches += hit
            if tok == self.eos_id or len(tokens) >= budget or not hit:
                break
        return tokens, matches

    def stream(self, prompt: Sequence[int], max_tokens: int) -> Iterator[list[int]]:
        """Yield the tokens of each round (EOS included as its last element when sampled). Generation ends after
        `max_tokens` tokens or at EOS. The cache is left as a plain loop leaves it: every emitted token but the last."""
        if max_tokens < 1:
            return
        st = self.stats
        n_prompt = len(prompt)
        out = self.target.prefill(prompt)
        tok = self.sampler.sample(out.logits[-1])
        st.emitted += 1
        yield [tok]
        count = 1
        hidden = out.hidden
        pos = n_prompt                                # position of `tok`
        while tok != self.eos_id and count < max_tokens:
            remaining = max_tokens - count
            draft = DraftOut([]) if remaining <= 1 else self.drafter.draft(tok, hidden, pos - 1)
            k = min(len(draft.tokens), self.num_draft, remaining - 1)
            out = self.target.verify([tok, *draft.tokens[:k]], pos)
            tokens, matches = self._walk(out, draft, k, remaining)
            kept = len(tokens)                        # positions pos .. pos + kept - 1 stay: tok + kept-1 drafts
            if kept < k + 1:
                self.target.truncate(pos + kept)
            if k:
                st.steps += 1
                st.proposed += k
                st.accepted += matches
                st.draft_len_hist[k] += 1
                st.accepted_hist[matches] += 1
            else:
                st.plain_steps += 1
            st.emitted += len(tokens)
            count += len(tokens)
            tok = tokens[-1]
            hidden = out.hidden[:kept]
            pos += kept
            yield tokens


def decode(target: Target, drafter: Drafter, prompt: Sequence[int], max_tokens: int, eos_id: int, *,
           sampler: Sampler | None = None, num_draft: int | None = None) -> tuple[list[int], DSparkDecoder]:
    """Whole completion, EOS cut off like `generate.py`. Returns (tokens, decoder); `decoder.stats` has the numbers."""
    dec = DSparkDecoder(target, drafter, eos_id=eos_id, sampler=sampler, num_draft=num_draft)
    out: list[int] = []
    for chunk in dec.stream(prompt, max_tokens):
        out.extend(chunk)
    if eos_id in out:
        out = out[:out.index(eos_id)]
    return out, dec
