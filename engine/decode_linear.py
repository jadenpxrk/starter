"""Decode dense backend and bounded, whole-step CUDA graph selection.

No Triton import on the CPU/native path. Calibration happens once per newly
captured shape, never inside graph capture or during replay. Timing logs go to
stderr. It compares real full-step graphs, not a repeatedly cache-hot matrix.
"""

import json
import math
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F


MAX_ROWS = 32
# Runtime guard against noisy/regressive choices, NOT the optimization objective.
MIN_GRAPH_GAIN = 0.10
# Soft check between policies; it cannot interrupt an in-progress JIT compile.
CALIBRATION_SECONDS = 90.0


def split_k_for(m, n, k, sms):
    """Add K parallelism only when output tiles alone underfill the device.

    No public-workload table: this is a matrix-shape/device occupancy heuristic,
    not a claim about measured occupancy or the optimal H100 configuration.
    """
    tiles = ((m + 15) // 16) * ((n + 63) // 64)
    target = max(1, (2 * sms + tiles - 1) // tiles)
    split = 1
    while split < target and split < 16 and split * 64 < k:
        split *= 2
    return split


def wins(reference_ms, candidate_ms):
    """Require improvement in every pair and >=10% median time reduction."""
    if len(reference_ms) != 3 or len(candidate_ms) != 3:
        return False
    if any(not math.isfinite(x) or x <= 0 for x in reference_ms + candidate_ms):
        return False
    return (all(c < r for r, c in zip(reference_ms, candidate_ms))
            and statistics.median(candidate_ms)
            <= (1 - MIN_GRAPH_GAIN) * statistics.median(reference_ms))


class DecodeLinearPlan:
    def __init__(self):
        self.active = False
        self.policy = "torch"
        self.audit = False
        self.audit_count = 0
        self.audit_max_abs = 0.0
        self.sms = None

    def linear(self, x, weight, bias=None):
        if (not self.active or self.policy == "torch" or bias is not None
                or torch.is_grad_enabled() or x.ndim != 3 or x.shape[1] != 1
                or not 1 <= x.shape[0] <= MAX_ROWS or not x.is_cuda
                or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
                or weight.device != x.device or not x.is_contiguous()
                or not weight.is_contiguous() or weight.ndim != 2
                or x.shape[-1] != weight.shape[1] or min(weight.shape) <= 0
                or (self.policy == "simt" and x.shape[0] != 1)):
            return F.linear(x, weight, bias)
        from kernels.decode_linear import small_linear

        if self.sms is None:
            # First use is eager warmup, before capture. No device-value read.
            self.sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        m, k, n = x.shape[0], x.shape[-1], weight.shape[0]
        out = small_linear(x.view(m, k), weight, self.policy,
                           split_k_for(m, n, k, self.sms)).view(m, 1, n)
        if self.audit:
            # Warmup-only operator check on the SAME candidate input at each
            # projection. Not a substitute for official own-prefix replay.
            ref = F.linear(x, weight)
            delta = (out.float() - ref.float()).abs()
            self.audit_count += 1
            self.audit_max_abs = max(self.audit_max_abs, float(delta.max()))
            if not bool((delta <= 0.002 + 0.01 * ref.float().abs()).all()):
                raise AssertionError(f"dense audit failed for M,N,K={m,n,k}; "
                                     f"max_abs={self.audit_max_abs}")
        return out

    def capture(self, state, tokens, position):
        """Compare at most two alternatives with the incumbent; keep two graphs.

        Restore inputs before EVERY replay. Only the next cache slot is
        overwritten; the prompt prefix is never edited. The chosen graph will
        overwrite that same slot on the real first step. No warmup answer is
        reused, and the original first output token/position are restored.
        """
        start = time.monotonic()
        state.linear_policy = "torch"
        best = state._capture_graph(tokens, position)
        best_policy = "torch"
        batch = state.shape[0]
        policies = (("simt", "tensorcore") if batch == 1
                    else ("tensorcore",) if 1 < batch <= MAX_ROWS else ())

        def restore():
            state.tokens.copy_(tokens)
            state.position.copy_(position)

        def measure(graph):
            # Identical restore-copy overhead in both policies. No .tolist(),
            # IPC, evolving prefix, or prefill is included in this diagnostic.
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(6):
                restore()
                graph.replay()
            end.record()
            end.synchronize()
            return begin.elapsed_time(end) / 6

        try:
            for policy in policies:
                if time.monotonic() - start >= CALIBRATION_SECONDS:
                    self._log(state, policy=policy, skipped="soft_calibration_budget")
                    break
                state.linear_policy = policy
                self.audit_count, self.audit_max_abs = 0, 0.0
                restore()
                try:
                    self.audit = True
                    state.step()  # Also compiles every actually used specialization.
                except AssertionError as exc:
                    # No graph is captured with a failed arithmetic audit. Do
                    # not catch CUDA runtime faults and pretend recovery is safe.
                    self._log(state, policy=policy, rejected="arithmetic_audit", detail=str(exc))
                    continue
                finally:
                    self.audit = False
                    restore()
                if self.audit_count == 0:
                    self._log(state, policy=policy, skipped="no_eligible_projections")
                    continue
                trial = state._capture_graph(tokens, position)
                incumbent_ms, candidate_ms = [], []
                for pair in range(3):
                    order = ((best, incumbent_ms), (trial, candidate_ms))
                    if pair % 2:
                        order = order[::-1]
                    for graph, samples in order:
                        samples.append(measure(graph))
                accept = wins(incumbent_ms, candidate_ms)
                self._log(state, incumbent=best_policy, candidate=policy,
                          incumbent_graph_ms=incumbent_ms, candidate_graph_ms=candidate_ms,
                          audited_projections=self.audit_count, audit_max_abs=self.audit_max_abs,
                          selected=policy if accept else best_policy)
                # measure() synchronized both graphs. Do not retain every graph
                # or share pools between graphs with different tensor lifetimes.
                if accept:
                    best.reset()
                    best, best_policy = trial, policy
                else:
                    trial.reset()
            return best, best_policy
        finally:
            restore()
            state.linear_policy = best_policy
            self._log(state, selected=best_policy, calibration_seconds=time.monotonic() - start)

    @staticmethod
    def _log(state, **data):
        print("DENSE_DECODE " + json.dumps({"shape": list(state.shape), **data},
                                          sort_keys=True, allow_nan=False),
              file=sys.stderr, flush=True)


class DecodeLinear(nn.Module):
    def __init__(self, reference, plan):
        super().__init__()
        self.weight = reference.weight
        self.bias = reference.bias
        self.in_features = reference.in_features
        self.out_features = reference.out_features
        self.plan = plan
        self.train(reference.training)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight, self.bias)
        return self.plan.linear(x, self.weight, self.bias)


def install_decode_linears(model):
    """Reuse ALL parameters, including the tied embedding/LM-head Parameter."""
    plan = DecodeLinearPlan()
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, DecodeLinear(getattr(layer.self_attn, name), plan))
        layer.mlp.linear_plan = plan
        layer.mlp.down_proj = DecodeLinear(layer.mlp.down_proj, plan)
    model.lm_head = DecodeLinear(model.lm_head, plan)
    model.decode_linear_plan = plan
    return plan
