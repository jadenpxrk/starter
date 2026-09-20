"""Exact-verification tests. CPU emulation is NOT native Flash or Triton execution."""

import ast
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))
import speculative as spec
import verify


def load_prefill_checks():
    target = importlib.util.spec_from_file_location("spec_prefill_checks", ROOT / "tests/test_prefill.py")
    module = importlib.util.module_from_spec(target)
    target.loader.exec_module(module)
    return module


checks = load_prefill_checks()


def method(name):
    tree = ast.parse((ENGINE / "decode.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeState")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(ENGINE / "decode.py"), "exec"), env)
    return env[name]


class VerifyKernelBody:
    """Run the actual kernel body with checked CPU pointers; NOT a GPU interpreter."""
    def __init__(self, omit_rotary_rounding=False):
        self.ops = checks.Ops()
        path = ENGINE / "kernels/verify.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_verify_qkv")
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        if omit_rotary_rounding:
            for n in node.body:
                if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id in ("left", "right")):
                    n.value = n.value.func.value.func.value
        env = {"tl": self.ops}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), env)
        self.body = env[node.name]

    def run(self, p, qw, kw, qe, ke, cos, sin, keys, values, length, start):
        b, nk, cap, d = keys.shape
        nq = p.shape[1] // d - 2 * nk
        q = torch.empty(b, length, nq, d, dtype=p.dtype)
        inputs = [checks.Pointer(x) for x in (p, qw, kw, cos, sin, start)]
        qp, kp, vp = [checks.Pointer(x) for x in (q, keys, values)]
        for row in range(b * length * (nq + nk)):
            self.ops.program = row
            self.body(*inputs, qp, kp, vp, length, cap, nq, nk, d, qe, ke)
        written = torch.zeros_like(keys, dtype=torch.int64)
        written[:, :, int(start):int(start) + length] = 1
        assert torch.equal(kp.writes.view_as(written), written)
        assert torch.equal(vp.writes.view_as(written), written)
        assert bool((qp.writes == 1).all())
        return q


def reference_qkv(p, qw, kw, qe, ke, cos, sin, keys, values, length, start):
    b, nk, _, d = keys.shape
    nq = p.shape[1] // d - 2 * nk
    q, k, v = p.reshape(b, length, -1).split((nq * d, nk * d, nk * d), -1)
    q = checks.norm(q.reshape(b, length, nq, d), qw, qe)
    k = checks.norm(k.reshape(b, length, nk, d), kw, ke)
    pos = int(start)
    def rotate(x):
        rotated = torch.cat((-x[..., d//2:], x[..., :d//2]), -1)
        return x * cos[None, pos:pos+length, None] + rotated * sin[None, pos:pos+length, None]
    q, k = rotate(q), rotate(k)
    keys[:, :, pos:pos+length].copy_(k.transpose(1, 2))
    values[:, :, pos:pos+length].copy_(v.reshape(b, length, nk, d).transpose(1, 2))
    return q.contiguous()


def cpu_native_block(q, k, v, cuq, cuk, width, cap, used, scale):
    """Varlen API semantics, not Flash implementation arithmetic."""
    out = torch.empty_like(q)
    for virtual in range(used.numel()):
        qs, ks, length = int(cuq[virtual]), int(cuk[virtual]), int(used[virtual])
        qa = q[qs:qs+width].transpose(0, 1).unsqueeze(0)
        ka, va = k[ks:ks+length].transpose(0, 1).unsqueeze(0), v[ks:ks+length].transpose(0, 1).unsqueeze(0)
        mask = torch.arange(length, device=q.device)[None, :] <= (
            length - width + torch.arange(width, device=q.device)[:, None])
        ya = F.scaled_dot_product_attention(qa, ka.expand(1, qa.shape[1], length, 128),
                                          va.expand(1, qa.shape[1], length, 128),
                                          attn_mask=mask, scale=scale)
        out[qs:qs+width] = ya[0].transpose(0, 1)
    return out


@contextmanager
def cpu_pipeline():
    with mock.patch.dict(sys.modules, {
        "kernels.verify": SimpleNamespace(verify_qkv=VerifyKernelBody().run),
        "kernels.decode_fused": SimpleNamespace(add_rms_norm=checks.add_norm, silu_mul=checks.silu_mul),
    }), mock.patch.object(verify.BlockContext, "attention", verify.BlockContext._views_and_call), \
            mock.patch.object(verify, "_native_block", side_effect=cpu_native_block):
        yield


def target(prefix):
    # Deliberately prefix-dependent; not just last-token transition.
    return (sum((i + 1) * int(x) for i, x in enumerate(prefix)) + len(prefix) * 3 + 7) % 31


class ToyState:
    """No GPU simulation. Actual emitter/commit logic with a finite causal oracle."""
    emit = method("emit")
    def __init__(self, prompts, count):
        b, t = len(prompts), len(prompts[0])
        self.cache = torch.full((b, t + max(count, 1) + 4), -999, dtype=torch.int64)
        self.cache[:, :t] = torch.tensor(prompts)
        self.tokens = torch.tensor([[target(p)] for p in prompts])
        self.position = torch.tensor([t])
        self.host = torch.empty_like(self.tokens)
        self.graph = self.ready = None
        self.steps = 0
    def step(self):
        pos = int(self.position)
        self.cache[:, pos:pos+1].copy_(self.tokens)
        self.tokens.copy_(torch.tensor([[target(row[:pos+1].tolist())] for row in self.cache]))
        self.position.add_(1)
        self.steps += 1


class ToyVerifier:
    commit = verify.VerificationPlan.commit
    def __init__(self, state):
        self.state, self.width, self.pending = state, spec.BLOCK, False
        self.calls = 0
    def enqueue(self, inputs):
        assert not self.pending
        pos = int(self.state.position)
        self.state.cache[:, pos:pos+self.width] = torch.tensor(inputs)
        self.predictions = torch.tensor([
            [target(row[:pos+i+1].tolist()) for i in range(self.width)] for row in self.state.cache])
        self.calls += 1
        self.pending = True
    def finish(self):
        self.pending = False
        return self.predictions.tolist()


def oracle_output(prompts, count):
    histories = deepcopy(prompts)
    rows = []
    for _ in range(count):
        row = [target(p) for p in histories]
        rows.append(row)
        for p, token in zip(histories, row):
            p.append(token)
    return rows


class SpeculativeCPU(unittest.TestCase):
    def test_suffix_index_matches_brute_force_and_bounds(self):
        rng = random.Random(104)
        index = spec.SuffixDraft([], window=23)
        for _ in range(300):
            index.append(rng.randrange(4))
            tokens = index.tokens
            expected = None
            for n in range(spec.MAX_N, spec.MIN_N - 1, -1):
                ends = [e for e in range(max(n, len(tokens)-index.window), len(tokens)-spec.DRAFT+1)
                        if tuple(tokens[e-n:e]) == tuple(tokens[-n:])]
                if len(tokens) >= n and ends:
                    expected = tokens[ends[-1]:ends[-1]+spec.DRAFT]
                    break
            self.assertEqual(index.propose(), expected)
            self.assertLessEqual(len(index.tokens), 2 * index.window)
            self.assertLessEqual(len(index.index), 4 * (index.window + 1))
        index = spec.SuffixDraft([1, 2, 3, 0, 9, 8, 1, 2, 3])
        self.assertEqual(index.propose(), [0, 9, 8])  # EOS-like ID 0 is ordinary.
        self.assertIsNone(spec.SuffixDraft([1, 2, 3]).propose())
        self.assertIsNone(spec.SuffixDraft(list(range(100))).propose())

    def test_acceptance_all_mismatch_positions_and_batch_intersection(self):
        for batch in (1, 3, 8):
            inputs = [[10+b, 20+b, 30+b, 40+b] for b in range(batch)]
            perfect = [[20+b, 30+b, 40+b, 50+b] for b in range(batch)]
            self.assertEqual(spec.accepted_count(inputs, perfect), 4)
            for fail_at in range(3):
                outputs = deepcopy(perfect)
                outputs[-1][fail_at] = -1
                self.assertEqual(spec.accepted_count(inputs, outputs), fail_at + 1)
        with self.assertRaises(ValueError):
            spec.accepted_count([[1]], [[2]])

    @torch.inference_mode()
    def test_actual_emitter_exact_oracle_sequence_rollback_and_batch_order(self):
        class ControlledDraft:
            mode = "perfect"
            lane = 0
            def __init__(self, tokens, **kwargs):
                self.tokens = list(tokens)
                self.lane = ControlledDraft.lane
                ControlledDraft.lane += 1
            def append(self, token):
                self.tokens.append(token)
            def propose(self):
                if self.mode == "none":
                    return None
                prefix = self.tokens.copy()
                proposal = []
                for _ in range(3):
                    value = target(prefix); proposal.append(value); prefix.append(value)
                if self.mode.startswith("wrong") and self.lane % 2 == 0:
                    j = int(self.mode[-1]); proposal[j] = (proposal[j] + 1) % 31
                return proposal
        for batch in (1, 3):
            prompts = [[b + 1, 5, 0, 7] for b in range(batch)]
            for count in (0, 1, 2, 5, 6, 12, 33):
                for mode in ("perfect", "none", "wrong0", "wrong1", "wrong2"):
                    state = ToyState(prompts, count)
                    verifier = ToyVerifier(state)
                    plan = spec.LookupPlan(state, verifier)
                    addresses = (state.tokens.data_ptr(), state.position.data_ptr(), state.cache.data_ptr())
                    ControlledDraft.lane, ControlledDraft.mode = 0, mode
                    with self.subTest(batch=batch, count=count, mode=mode), \
                            mock.patch.object(spec, "SuffixDraft", ControlledDraft):
                        actual = list(plan.emit(prompts, count))
                        self.assertEqual(actual, oracle_output(prompts, count))
                        self.assertEqual(len(actual), count)
                        self.assertTrue(all(len(row) == batch and all(type(x) is int for x in row) for row in actual))
                        expected_pos = len(prompts[0]) + max(0, count - 1)
                        self.assertEqual(int(state.position), expected_pos)
                        if count:
                            cached = torch.cat((torch.tensor(prompts), torch.tensor(actual[:-1], dtype=torch.int64).reshape(count-1, batch).T), 1)
                            torch.testing.assert_close(state.cache[:, :expected_pos], cached, rtol=0, atol=0)
                        self.assertEqual(addresses, (state.tokens.data_ptr(), state.position.data_ptr(), state.cache.data_ptr()))
                        if count == 33 and mode == "perfect":
                            self.assertLess(state.steps + verifier.calls, count - 1)
                        if mode == "wrong0":
                            self.assertLessEqual(verifier.calls, spec.MAX_UNPRODUCTIVE)
        # Negative control: a later prediction after a first-position mismatch
        # is NOT necessarily valid on the corrective token's own prefix.
        prefix, inp = [3, 8, 1], [7, 30, 29, 28]
        pred = [target(prefix + inp[:i+1]) for i in range(4)]
        self.assertNotEqual(inp[1], pred[0])
        self.assertNotEqual(pred[1], target(prefix + [inp[0], pred[0]]))

    @torch.inference_mode()
    def test_real_suffix_proposals_no_cross_request_history(self):
        prompts = [[1, 2, 3, 0, 9, 8, 1, 2, 3], [4, 5, 6, 0, 9, 8, 4, 5, 6]]
        for _ in range(2):
            state = ToyState(prompts, 24)
            verifier = ToyVerifier(state)
            actual = list(spec.LookupPlan(state, verifier).emit(prompts, 24))
            self.assertEqual(actual, oracle_output(prompts, 24))
            self.assertFalse(verifier.pending)
        self.assertEqual(prompts[0], [1, 2, 3, 0, 9, 8, 1, 2, 3])

    @torch.inference_mode()
    def test_kernel_body_absolute_positions_writes_and_cast_negative_control(self):
        torch.manual_seed(71)
        mismatches = 0
        for batch, nq, nk, cap, start in ((1, 32, 8, 17, 0), (1, 32, 8, 544, 512),
                                         (3, 12, 3, 39, 35), (16, 32, 8, 9, 3)):
            p = torch.randn(batch * 4, (nq + 2 * nk) * 128).bfloat16()
            qw, kw = torch.randn(128).bfloat16(), torch.randn(128).bfloat16()
            cos, sin = checks.angles(cap)
            k = torch.full((batch, nk, cap, 128), float('nan'), dtype=torch.bfloat16)
            v = k.clone(); kr, vr = k.clone(), v.clone()
            position = torch.tensor([start])
            before = [x.clone() for x in (p, qw, kw, cos, sin, position)]
            actual = VerifyKernelBody().run(p, qw, kw, 1e-6, 3e-6, cos, sin, k, v, 4, position)
            expected = reference_qkv(p, qw, kw, 1e-6, 3e-6, cos, sin, kr, vr, 4, position)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(k, kr, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(v, vr, rtol=0, atol=0, equal_nan=True)
            for x, old in zip((p, qw, kw, cos, sin, position), before):
                torch.testing.assert_close(x, old, rtol=0, atol=0)
            bad = VerifyKernelBody(True).run(p, qw, kw, 1e-6, 3e-6, cos, sin, kr, vr, 4, position)
            mismatches += int((bad != actual).sum())
        self.assertGreater(mismatches, 0)

    @torch.inference_mode()
    def test_varlen_mapping_causality_lengths_schema_and_negative_controls(self):
        torch.manual_seed(27)
        for batch, nq, nk, cap, start in ((1, 32, 8, 8, 0), (3, 12, 3, 39, 27), (2, 8, 2, 15, 11)):
            ctx = verify.BlockContext(batch, nq, nk, cap, 4, "cpu")
            pointers = [t.data_ptr() for t in (ctx.cu_q, ctx.cu_k, ctx.used)]
            q, k = torch.randn(batch, 4, nq, 128).bfloat16(), torch.randn(batch, nk, cap, 128).bfloat16()
            v = torch.randn_like(k)
            k[:, :, start+4:] = float('nan'); v[:, :, start+4:] = float('nan')
            ctx.prepare(torch.tensor([start]))
            with mock.patch.object(verify, "_native_block", side_effect=cpu_native_block) as op:
                actual = ctx._views_and_call(q, k, v, 128**-0.5)
            args = op.call_args.args
            self.assertEqual(args[1].data_ptr(), k.data_ptr())
            self.assertEqual(args[2].data_ptr(), v.data_ptr())
            length = start + 4
            mask = torch.arange(length)[None, :] <= start + torch.arange(4)[:, None]
            expected = F.scaled_dot_product_attention(q.transpose(1, 2),
                k[:, :, :length].repeat_interleave(nq//nk, 1),
                v[:, :, :length].repeat_interleave(nq//nk, 1), attn_mask=mask, scale=128**-0.5)
            torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=0.02, atol=0.002)
            self.assertTrue(bool(actual.isfinite().all()))
            if start:
                wrong = F.scaled_dot_product_attention(q.transpose(1, 2),
                    k[:, :, :length].repeat_interleave(nq//nk, 1),
                    v[:, :, :length].repeat_interleave(nq//nk, 1), is_causal=True)
                self.assertFalse(torch.equal(expected, wrong))
            if length < cap:
                # Using physical capacity instead of valid length exposes poison.
                ctx.used.fill_(cap)
                with mock.patch.object(verify, "_native_block", side_effect=cpu_native_block):
                    bad = ctx._views_and_call(q, k, v, 128**-0.5)
                self.assertFalse(bool(bad.isfinite().all()))
            ctx.prepare(torch.tensor([0]))
            self.assertTrue(bool((ctx.used == 4).all()))
            self.assertEqual(pointers, [t.data_ptr() for t in (ctx.cu_q, ctx.cu_k, ctx.used)])
        schema = torch.ops.aten._flash_attention_forward.default._schema
        self.assertEqual([a.name for a in schema.arguments][:10],
                         ["query", "key", "value", "cum_seq_q", "cum_seq_k", "max_q", "max_k", "dropout_p", "is_causal", "return_debug_mask"])
        x = torch.empty(1)
        with mock.patch.object(torch.ops.aten._flash_attention_forward, "default", return_value=(x,)) as op:
            self.assertIs(verify._native_block(x, x, x, x, x, 4, 39, x, 0.2), x)
        self.assertEqual(op.call_args.args[5:], (4, 39, 0.0, True, False))
        self.assertIs(op.call_args.kwargs['seqused_k'], x)

    @torch.inference_mode()
    def test_full_block_logits_on_each_own_input_prefix_and_rollback(self):
        torch.manual_seed(51)
        for batch, length in ((1, 3), (2, 7), (3, 9)):
            model, cache, (cos, sin) = checks.fixture(batch, length, extra=12)
            ids = torch.randint(0, 127, (batch, length))
            checks.materialized(model, cache, ids, cos, sin)
            ctx = verify.BlockContext(batch, 4, 2, cache.max_cache_len, 4, "cpu")
            block = torch.randint(0, 127, (batch, 4))
            start = torch.tensor([length])
            pointers = [t.data_ptr() for t in cache.key_cache + cache.value_cache]
            with cpu_pipeline():
                logits = verify.verify_forward(model, cache, block, start, ctx, cos, sin)
            reference_cache = deepcopy(cache)
            for i in range(4):
                own = torch.cat((ids, block[:, :i+1]), 1)
                expected = checks.materialized(model, reference_cache, own, cos, sin)
                torch.testing.assert_close(logits[:, i:i+1], expected, rtol=0.03, atol=0.04)
            # Roll back two inputs, keeping the candidate's corrected next ID.
            accepted = 2
            next_ids = logits[:, accepted-1].argmax(-1, keepdim=True)
            next_block = torch.cat((next_ids, torch.randint(0, 127, (batch, 3))), 1)
            start.add_(accepted)
            with cpu_pipeline():
                second = verify.verify_forward(model, cache, next_block, start, ctx, cos, sin)
            for i in range(4):
                own = torch.cat((ids, block[:, :accepted], next_block[:, :i+1]), 1)
                expected = checks.materialized(model, reference_cache, own, cos, sin)
                torch.testing.assert_close(second[:, i:i+1], expected, rtol=0.03, atol=0.04)
            self.assertEqual(pointers, [t.data_ptr() for t in cache.key_cache + cache.value_cache])
            valid = int(start) + 4
            for a, b in zip(cache.key_cache + cache.value_cache, reference_cache.key_cache + reference_cache.value_cache):
                torch.testing.assert_close(a[:, :, :valid], b[:, :, :valid], rtol=0.03, atol=0.04)

    @torch.inference_mode()
    def test_verifier_capture_replay_commit_order_with_fake_cuda(self):
        # This executes the real wrapper lifecycle, not CUDA graph semantics.
        events = []
        class Stream:
            def wait_stream(self, other): events.append("wait_stream")
        class Event:
            def record(self): events.append("record")
            def synchronize(self): events.append("synchronize")
        plan = object.__new__(verify.VerificationPlan)
        plan.width = 4
        plan.state = SimpleNamespace(tokens=torch.tensor([[9], [11]]), position=torch.tensor([7]))
        plan.inputs = torch.empty(2, 4, dtype=torch.int64)
        plan.input_host, plan.host = torch.empty_like(plan.inputs), torch.empty_like(plan.inputs)
        plan.graph, plan.predictions, plan.pending = None, None, False
        plan.ready = Event()
        def call():
            events.append("call")
            return plan.inputs + plan.state.position
        plan._call = call
        class Graph:
            def replay(self):
                events.append("replay")
                plan.predictions.copy_(call())
        @contextmanager
        def region(*args, **kwargs):
            self.assertNotIn("pool", kwargs)
            yield
        with mock.patch.object(torch.cuda, "current_stream", return_value=Stream()), \
                mock.patch.object(torch.cuda, "Stream", return_value=Stream()), \
                mock.patch.object(torch.cuda, "stream", side_effect=region), \
                mock.patch.object(torch.cuda, "graph", side_effect=region), \
                mock.patch.object(torch.cuda, "CUDAGraph", side_effect=Graph):
            plan.capture()
            self.assertEqual(events.count("call"), 3)
            self.assertEqual(int(plan.state.position), 7)
            torch.testing.assert_close(plan.state.tokens, torch.tensor([[9], [11]]))
            pointers = (plan.inputs.data_ptr(), plan.predictions.data_ptr())
            plan.capture()
            self.assertEqual(events.count("call"), 3)
            for offset in (0, 4):
                rows = [[1+offset, 2, 3, 4], [5, 6, 7, 8+offset]]
                before = int(plan.state.position)
                plan.enqueue(rows)
                with self.assertRaises(RuntimeError): plan.enqueue(rows)
                with self.assertRaises(ValueError): plan.commit(1)
                actual = plan.finish()
                self.assertEqual(actual, (torch.tensor(rows) + before).tolist())
                self.assertEqual(events[-1], "synchronize")
                plan.commit(2)
                self.assertEqual(int(plan.state.position), before + 2)
                torch.testing.assert_close(plan.state.tokens, plan.predictions[:, 1:2])
                self.assertEqual(pointers, (plan.inputs.data_ptr(), plan.predictions.data_ptr()))
            with self.assertRaises(RuntimeError): plan.finish()

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None,
                         "real Transformers CPU integration requires the installed package")
    @torch.inference_mode()
    def test_real_transformers_cpu_block_and_logical_rollback(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache
        from mlp import PackedMLP
        torch.manual_seed(193)
        cfg = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=96,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
            max_position_embeddings=1024, rope_theta=5_000_000, tie_word_embeddings=True,
            sliding_window=None)
        cfg._attn_implementation = "sdpa"
        native = Qwen3ForCausalLM(cfg).eval().bfloat16()
        candidate = deepcopy(native)
        for layer in candidate.model.layers:
            a = layer.self_attn
            a.qkv_weight = torch.cat((a.q_proj.weight, a.k_proj.weight, a.v_proj.weight))
            layer.mlp = PackedMLP(layer.mlp)
        for batch, length in ((1, 3), (2, 7)):
            cap = length + 12
            cache = StaticCache(cfg, max_batch_size=batch, max_cache_len=cap,
                                device="cpu", dtype=torch.bfloat16)
            cos, sin = [t[0].contiguous() for t in native.model.rotary_emb(
                torch.empty(0, dtype=torch.bfloat16), torch.arange(cap).unsqueeze(0))]
            context = verify.BlockContext(batch, 4, 2, cap, 4, "cpu")
            for attempt in range(2):
                prompt = torch.randint(0, 127, (batch, length))
                prefix = native(input_ids=prompt, use_cache=True).past_key_values
                for k, v, pk, pv in zip(cache.key_cache, cache.value_cache, prefix.key_cache, prefix.value_cache):
                    k.fill_(33); v.fill_(-41)
                    k[:, :, :length].copy_(pk); v[:, :, :length].copy_(pv)
                block = torch.randint(0, 127, (batch, 4))
                start = torch.tensor([length])
                with cpu_pipeline():
                    logits = verify.verify_forward(candidate, cache, block, start, context, cos, sin)
                for i in range(4):
                    own = torch.cat((prompt, block[:, :i+1]), 1)
                    expected = native(input_ids=own, use_cache=False).logits[:, -1:]
                    torch.testing.assert_close(logits[:, i:i+1], expected, rtol=0.03, atol=0.04)
                # Corrective token is not cached until the next forward.
                next_block = torch.cat((logits[:, 1].argmax(-1, keepdim=True),
                                        torch.randint(0, 127, (batch, 3))), 1)
                start.add_(2)
                with cpu_pipeline():
                    logits = verify.verify_forward(candidate, cache, next_block, start, context, cos, sin)
                own = torch.cat((prompt, block[:, :2], next_block), 1)
                expected = native(input_ids=own, use_cache=False).logits[:, -4:]
                torch.testing.assert_close(logits, expected, rtol=0.03, atol=0.04)

    def test_factory_skips_unsupported_and_short_generation(self):
        state = SimpleNamespace(prefill_plan=object(), flash_context=object(), fused_forward=object(), shape=(3, 17, 9))
        with mock.patch.object(verify, "VerificationPlan", return_value=object()) as constructor:
            self.assertIsInstance(spec.make_lookup_plan(state), spec.LookupPlan)
            constructor.assert_called_once_with(state, spec.BLOCK)
        state.shape = (3, 17, 5)
        self.assertIsNone(spec.make_lookup_plan(state))
        state.shape = (spec.MAX_HISTORY_TOKENS, 17, 9)
        self.assertIsNone(spec.make_lookup_plan(state))
        state.shape = (3, 17, 9); state.prefill_plan = None
        self.assertIsNone(spec.make_lookup_plan(state))


HAS_CUDA = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_CUDA, "CUDA/Triton required; CPU emulation does not execute these kernels")
class SpeculativeCUDA(unittest.TestCase):
    @torch.inference_mode()
    def test_actual_qkv_and_native_varlen_causal_attention(self):
        from kernels.verify import verify_qkv
        torch.manual_seed(89)
        for b, cap, start in ((1, 544, 512), (4, 2080, 2048), (16, 640, 512), (3, 51, 47)):
            p = torch.randn(b * 4, 6144, device="cuda", dtype=torch.bfloat16)
            qw, kw = [torch.randn(128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            c, s = checks.angles(cap, "cuda")
            k = torch.randn(b, 8, cap, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k); k[:, :, start+4:] = float('nan'); v[:, :, start+4:] = float('nan')
            kr, vr = k.clone(), v.clone()
            pos = torch.tensor([start], device="cuda")
            q = verify_qkv(p, qw, kw, 1e-6, 3e-6, c, s, k, v, 4, pos)
            qr = reference_qkv(p, qw, kw, 1e-6, 3e-6, c, s, kr, vr, 4, pos)
            torch.testing.assert_close(q, qr, rtol=0.02, atol=0.02)
            torch.testing.assert_close(k, kr, rtol=0.02, atol=0.02, equal_nan=True)
            torch.testing.assert_close(v, vr, rtol=0, atol=0, equal_nan=True)
            context = verify.BlockContext(b, 32, 8, cap, 4, "cuda")
            context.prepare(pos)
            actual = context.attention(q, k, v, 128**-0.5)
            mask = torch.arange(start+4, device="cuda")[None] <= start + torch.arange(4, device="cuda")[:, None]
            expected = F.scaled_dot_product_attention(q.transpose(1, 2), k[:, :, :start+4].repeat_interleave(4, 1),
                v[:, :, :start+4].repeat_interleave(4, 1), attn_mask=mask, scale=128**-0.5).transpose(1, 2)
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
            self.assertTrue(bool(actual.isfinite().all()))

    @torch.inference_mode()
    def test_engine_graphs_forced_verification_reuse_own_prefix_and_eos(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from engine import Engine
        cfg = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=96,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
            max_position_embeddings=1024, rope_theta=5_000_000, tie_word_embeddings=True,
            sliding_window=None, eos_token_id=0)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(37)
        model = Qwen3ForCausalLM(cfg).eval()
        class ForcedDraft:
            def __init__(self, tokens, **kwargs): pass
            def append(self, token): pass
            def propose(self): return [0, 0, 0]
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            engine = Engine(path)
            native = model.cuda().bfloat16()
            prior_shape, prior_pointers = None, None
            for b, length, count in ((1, 7, 17), (1, 7, 17), (3, 11, 17), (3, 11, 1), (3, 11, 0)):
                prompt = torch.randint(0, 127, (b, length), device="cuda")
                with mock.patch.object(spec, "SuffixDraft", ForcedDraft), \
                        mock.patch.object(verify.VerificationPlan, "enqueue", autospec=True,
                                          side_effect=verify.VerificationPlan.enqueue) as enqueued:
                    rows = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(rows), count)
                self.assertTrue(all(len(row) == b and all(type(t) is int for t in row) for row in rows))
                if count >= 6:
                    self.assertGreater(enqueued.call_count, 0)
                if not count: continue
                emitted = torch.tensor(rows, device="cuda").T
                own = torch.cat((prompt, emitted[:, :-1]), 1)
                logits = native(input_ids=own, use_cache=False).logits[:, length-1:]
                gap = logits.float().amax(-1) - logits.float().gather(-1, emitted.unsqueeze(-1)).squeeze(-1)
                self.assertTrue(bool(gap.isfinite().all()))
                self.assertLessEqual(float(gap.max()), 2.0)
                self.assertEqual(int(engine.decode_state.position), length + count - 1)
                state = engine.decode_state
                ptrs = [t.data_ptr() for t in state.cache.key_cache + state.cache.value_cache]
                if prior_shape == state.shape:
                    self.assertEqual(ptrs, prior_pointers)
                prior_shape, prior_pointers = state.shape, ptrs
                for tensor in state.cache.key_cache + state.cache.value_cache:
                    tensor.fill_(float("nan"))
            # A genuine EOS argmax may repeat; no early stop in either path.
            for p in native.parameters(): p.zero_()
            native.cpu().save_pretrained(path)
            engine = Engine(path)
            rows = list(engine.generate([[0] * 7, [0] * 7], 13))
            self.assertEqual(rows, [[0, 0]] * 13)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
