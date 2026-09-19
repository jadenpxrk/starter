from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


spec = importlib.util.spec_from_file_location(
    'decode', Path(__file__).resolve().parents[1] / 'engine/decode.py',
)
decode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode)

spec = importlib.util.spec_from_file_location(
    'decode_attention', Path(__file__).resolve().parents[1] / 'engine/decode_attention.py',
)
decode_attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode_attention)


def tiny_model(dtype=torch.float32):
    config = Qwen3Config(
        vocab_size=127, hidden_size=48, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, rope_theta=5_000_000,
        tie_word_embeddings=True, sliding_window=None,
    )
    config._attn_implementation = 'sdpa'
    return Qwen3ForCausalLM(config).eval().to(dtype=dtype)


class DecodeTests(unittest.TestCase):
    def test_cached_logits_match_native_across_prompts(self):
        torch.manual_seed(0)
        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16):
                model = tiny_model(dtype)
                candidate = deepcopy(model)
                decode_attention.install_decode_attention(candidate)
                self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'], sdpa_attention_forward)
                self.assertEqual(model.config._attn_implementation, 'sdpa')
                for layer in candidate.model.layers:
                    self.assertIs(
                        ALL_ATTENTION_FUNCTIONS[layer.self_attn.config._attn_implementation],
                        decode_attention.grouped_decode_attention,
                    )
                for batch, length, count in ((1, 1, 2), (2, 7, 5), (4, 13, 1)):
                    state = decode.DecodeState(candidate, batch, length, count)
                    buffers = state.cache.key_cache + state.cache.value_cache
                    addresses = [tensor.data_ptr() for tensor in buffers]
                    for attempt in range(2):
                        with self.subTest(dtype=dtype, batch=batch, length=length,
                                          count=count, attempt=attempt):
                            # Unused capacity must remain invisible, even after reuse.
                            for tensor in buffers:
                                tensor.fill_(33)
                            prompt = torch.randint(0, 127, (batch, length))
                            actual = state.prefill(prompt)
                            current, cache = prompt, None
                            for step in range(count):
                                expected = model(
                                    input_ids=current, past_key_values=cache,
                                    use_cache=True, logits_to_keep=1,
                                )
                                tolerance = (
                                    dict(rtol=2e-2, atol=2e-3) if dtype == torch.bfloat16
                                    else dict(rtol=1e-5, atol=1e-6)
                                )
                                torch.testing.assert_close(actual, expected.logits, **tolerance)
                                torch.testing.assert_close(
                                    state.tokens, expected.logits[:, -1].argmax(-1, keepdim=True),
                                )
                                current = state.tokens.clone()
                                cache = expected.past_key_values
                                if step + 1 < count:
                                    actual = state.step()
                            self.assertEqual(state.position.item(), length + count - 1)
                            self.assertEqual(addresses, [tensor.data_ptr() for tensor in buffers])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_engine_graph_replay_matches_native(self):
        if importlib.util.find_spec('triton') is None:
            self.skipTest('Triton is required')
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
        try:
            from engine import Engine
        finally:
            sys.path.pop(0)

        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as checkpoint, torch.inference_mode():
            model = tiny_model()
            model.save_pretrained(checkpoint)
            engine = Engine(checkpoint)
            model = model.cuda().bfloat16()
            for batch, length, count in ((2, 7, 5), (2, 7, 5), (1, 1, 2), (2, 7, 1), (2, 7, 0)):
                prompt = torch.randint(0, 127, (batch, length), device='cuda')
                current, cache, expected = prompt, None, []
                for _ in range(count):
                    result = model(input_ids=current, past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    current = result.logits[:, -1].argmax(-1, keepdim=True)
                    cache = result.past_key_values
                    expected.append(current[:, 0].tolist())
                self.assertEqual(list(engine.generate(prompt.tolist(), count)), expected)


if __name__ == '__main__':
    unittest.main()
