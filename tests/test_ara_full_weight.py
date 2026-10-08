# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
from isolated_settings import IsolatedSettings
from transformers import Qwen2Config, Qwen2ForCausalLM

from heretic.model import ARAParameters, Model


def make_model(dtype: torch.dtype) -> Model:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    tiny = Qwen2ForCausalLM(config).to(dtype).eval()
    model = Model.__new__(Model)
    model.settings = IsolatedSettings(
        model="tiny", use_ara=True, target_components=["mlp.down_proj"]
    )
    model.model = tiny
    return model


def synthetic_io(model: Model, seed: int, shift: float):
    """Captured-looking I/O for layer 0's down_proj: inputs of shape (prompts, 32)."""
    torch.manual_seed(seed)
    weight = model.get_layer_modules(0)["mlp.down_proj"][0].weight.detach().float()
    inputs = torch.randn(40, weight.shape[1]) + shift
    outputs = inputs @ weight.T
    return [{"mlp.down_proj": {0: (inputs, outputs)}}, {}]


class FullWeightARATests(unittest.TestCase):
    def test_edits_a_bf16_matrix_and_keeps_its_dtype(self) -> None:
        model = make_model(torch.bfloat16)
        module = model.get_layer_modules(0)["mlp.down_proj"][0]
        before = module.weight.detach().float().clone()

        model.ara_abliterate(
            synthetic_io(model, seed=1, shift=0.0),
            synthetic_io(model, seed=2, shift=2.0),
            ARAParameters(
                start_layer_index=0,
                end_layer_index=1,
                preserve_good_behavior_weight=0.3,
                steer_bad_behavior_weight=0.01,
                overcorrect_relative_weight=1.0,
                neighbor_count=3,
            ),
        )

        after = module.weight.detach().float()
        self.assertEqual(module.weight.dtype, torch.bfloat16)
        self.assertGreater((after - before).norm().item(), 0.0)
        # Row-magnitude preservation: row norms stay what they were.
        torch.testing.assert_close(
            after.norm(dim=1), before.norm(dim=1), rtol=1e-2, atol=1e-2
        )

    def test_untargeted_layer_is_left_untouched(self) -> None:
        model = make_model(torch.bfloat16)
        other = model.get_layer_modules(1)["mlp.down_proj"][0]
        before = other.weight.detach().clone()

        model.ara_abliterate(
            synthetic_io(model, seed=1, shift=0.0),
            synthetic_io(model, seed=2, shift=2.0),
            ARAParameters(
                start_layer_index=0,
                end_layer_index=1,
                preserve_good_behavior_weight=0.3,
                steer_bad_behavior_weight=0.01,
                overcorrect_relative_weight=1.0,
                neighbor_count=3,
            ),
        )

        self.assertTrue(torch.equal(other.weight.detach(), before))


if __name__ == "__main__":
    unittest.main()
