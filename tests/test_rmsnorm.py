import importlib.util
from pathlib import Path
import sys
import unittest


class RMSNormTests(unittest.TestCase):
    def test_matches_qwen_bf16_normalization(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('PyTorch is required')
        import torch

        if not torch.cuda.is_available() or importlib.util.find_spec('triton') is None:
            self.skipTest('CUDA and Triton are required')
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
        try:
            from engine import FusedRMSNorm
        finally:
            sys.path.pop(0)

        torch.manual_seed(0)
        with torch.inference_mode():
            for width in (128, 2560):
                reference = Qwen3RMSNorm(width, eps=1e-6).cuda().bfloat16()
                reference.weight.normal_()
                fused = FusedRMSNorm(reference)
                for batch, length in ((1, 1), (4, 1), (16, 1), (4, 2048), (16, 512)):
                    for noncontiguous in (False, True):
                        with self.subTest(width=width, batch=batch, length=length,
                                          noncontiguous=noncontiguous):
                            x = torch.randn(batch, length, width, device='cuda',
                                            dtype=torch.bfloat16)
                            if noncontiguous:
                                x = x.transpose(0, 1)
                            torch.testing.assert_close(
                                fused(x), reference(x), rtol=8e-3, atol=1e-5,
                            )


if __name__ == '__main__':
    unittest.main()
