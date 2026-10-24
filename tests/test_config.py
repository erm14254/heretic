# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from isolated_settings import IsolatedSettings
from pydantic import ValidationError

from heretic.config import ARASearchSpace, ScorerConfig, Settings


class ScorerConfigTests(unittest.TestCase):
    def test_accepts_slug_like_instance_name(self) -> None:
        config = ScorerConfig(
            plugin="heretic.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
            instance_name="small-1",
        )

        self.assertEqual(config.instance_name, "small-1")

    def test_rejects_empty_instance_name(self) -> None:
        with self.assertRaises(ValidationError):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name=" \t",
            )

    def test_rejects_whitespace_in_instance_name(self) -> None:
        for instance_name in ["small name", "small\tname", "small\nname"]:
            with self.subTest(instance_name=instance_name):
                with self.assertRaisesRegex(
                    ValidationError, "whitespace is not allowed"
                ):
                    ScorerConfig(
                        plugin="heretic.scorers.keyword_rate.KeywordRate",
                        optimization="minimize",
                        instance_name=instance_name,
                    )

    def test_rejects_dot_in_instance_name(self) -> None:
        with self.assertRaisesRegex(ValidationError, "'\\.' is not allowed"):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name="small.name",
            )


class TargetComponentsTests(unittest.TestCase):
    def test_defaults_to_all_components(self) -> None:
        settings = IsolatedSettings(**{"model": "test-model"})

        self.assertEqual(
            settings.target_components,
            ["attn.o_proj", "attn.out_proj", "mlp.down_proj"],
        )

    def test_accepts_subset(self) -> None:
        settings = IsolatedSettings(
            **{"model": "test-model", "target_components": ["attn.o_proj"]}
        )

        self.assertEqual(settings.target_components, ["attn.o_proj"])

    def test_drops_duplicates(self) -> None:
        settings = IsolatedSettings(
            **{
                "model": "test-model",
                "target_components": ["mlp.down_proj", "attn.o_proj", "mlp.down_proj"],
            }
        )

        self.assertEqual(settings.target_components, ["mlp.down_proj", "attn.o_proj"])

    def test_accepts_linear_attention_component(self) -> None:
        settings = IsolatedSettings(
            model="test-model", target_components=["attn.out_proj"]
        )

        self.assertEqual(settings.target_components, ["attn.out_proj"])

    def test_rejects_unknown_component(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unknown component"):
            IsolatedSettings(
                **{"model": "test-model", "target_components": ["attn.q_proj"]}
            )

    def test_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one component"):
            IsolatedSettings(**{"model": "test-model", "target_components": []})


class ARASearchSpaceTests(unittest.TestCase):
    def test_defaults_match_the_original_ranges(self) -> None:
        space = ARASearchSpace()

        self.assertEqual(space.layer_bounds(64), (0, 32, 33, 64))
        self.assertEqual(
            (
                space.preserve_good_behavior_weight_min,
                space.preserve_good_behavior_weight_max,
            ),
            (0.0, 1.0),
        )
        self.assertEqual(
            (
                space.overcorrect_relative_weight_min,
                space.overcorrect_relative_weight_max,
            ),
            (0.0, 1.3),
        )
        self.assertEqual((space.neighbor_count_min, space.neighbor_count_max), (1, 15))

    def test_layer_bounds_use_explicit_values_and_clamp_to_the_model(self) -> None:
        space = ARASearchSpace(start_layer_index_max=8, end_layer_index_min=56)

        self.assertEqual(space.layer_bounds(64), (0, 8, 56, 64))
        # Bounds written for a bigger model are clamped and kept ordered.
        self.assertEqual(space.layer_bounds(32), (0, 8, 32, 32))

    def test_rejects_inverted_range(self) -> None:
        with self.assertRaisesRegex(ValidationError, "neighbor_count_max must be >="):
            ARASearchSpace(neighbor_count_min=10, neighbor_count_max=5)

    def test_settings_accept_a_narrowed_space(self) -> None:
        settings = IsolatedSettings(
            model="test-model",
            ara_search_space={
                "overcorrect_relative_weight_min": 1.0,
                "overcorrect_relative_weight_max": 1.2,
            },
        )

        self.assertEqual(settings.ara_search_space.overcorrect_relative_weight_min, 1.0)
        self.assertEqual(settings.ara_search_space.overcorrect_relative_weight_max, 1.2)
        self.assertEqual(settings.ara_search_space.neighbor_count_max, 15)


class ModelValidateInitOnlyTests(unittest.TestCase):
    """Port of p-e-w #484's tests; plugin tables are [scorer.X] on this fork."""

    def validate(
        self, values: dict, config_toml: str, argv: tuple[str, ...] = ("heretic",)
    ) -> Settings:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(config_toml)
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                with patch.object(sys, "argv", list(argv)):
                    return Settings.model_validate_init_only(
                        values, Settings(model="current")
                    )
            finally:
                os.chdir(previous_directory)

    def test_keeps_plugin_tables_from_values(self) -> None:
        settings = self.validate(
            {
                "model": "model",
                "scorer": {"KeywordRate": {"print_responses": True}},
            },
            'save_directory = "toml"\n[scorer.KeywordRate]\nscore_name = "from toml"\n',
            ("heretic", "--save-directory", "current"),
        )

        self.assertEqual(
            settings.model_extra,
            {"scorer": {"KeywordRate": {"print_responses": True}}},
        )
        # Settings with exclude=True come from the current settings.
        self.assertEqual(settings.save_directory, "current")

    def test_drops_plugin_tables_missing_from_values(self) -> None:
        settings = self.validate(
            {"model": "model"},
            "[scorer.KeywordRate]\nprint_responses = true\n",
        )

        self.assertEqual(settings.model_extra, {})

    def test_other_sources_do_not_override_values(self) -> None:
        values = {
            "model": "model",
            "n_trials": 50,
            "max_response_length": 60,
            "n_startup_trials": 20,
        }

        with patch.dict(os.environ, {"HERETIC_MAX_RESPONSE_LENGTH": "9"}):
            settings = self.validate(
                values,
                "n_trials = 7\n",
                ("heretic", "--n-startup-trials", "3"),
            )

        for name, value in values.items():
            self.assertEqual(getattr(settings, name), value)

    def test_settings_missing_from_values_get_defaults(self) -> None:
        with patch.dict(os.environ, {"HERETIC_MAX_RESPONSE_LENGTH": "9"}):
            settings = self.validate(
                {"model": "model"},
                "n_trials = 7\n",
                ("heretic", "--n-startup-trials", "3"),
            )

        for name in ["n_trials", "max_response_length", "n_startup_trials"]:
            self.assertEqual(
                getattr(settings, name), Settings.model_fields[name].default
            )

    def test_keeps_top_level_tables_from_values(self) -> None:
        settings = self.validate(
            {"model": "model", "max_memory": {"0": "20GiB"}},
            'max_memory = { "1" = "8GiB" }\n',
        )

        self.assertEqual(settings.max_memory, {"0": "20GiB"})

    def test_resumed_search_space_ignores_edited_config(self) -> None:
        # A study resumed after config.toml's search ranges were edited keeps the
        # ranges it was started with, key for key.
        settings = self.validate(
            {
                "model": "model",
                "ara_search_space": {"overcorrect_relative_weight_max": 1.05},
            },
            "[ara_search_space]\n"
            "overcorrect_relative_weight_max = 1.3\n"
            "neighbor_count_max = 64\n",
        )

        self.assertEqual(
            settings.ara_search_space.overcorrect_relative_weight_max, 1.05
        )
        self.assertEqual(
            settings.ara_search_space.neighbor_count_max,
            ARASearchSpace.model_fields["neighbor_count_max"].default,
        )


if __name__ == "__main__":
    unittest.main()
