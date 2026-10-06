# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from heretic.scorers.benchmark_score import Settings as BenchmarkSettings


class BenchmarkSettingsTests(unittest.TestCase):
    def test_defaults_keep_the_raw_harness_behavior(self) -> None:
        settings = BenchmarkSettings()

        self.assertFalse(settings.apply_chat_template)
        self.assertIsNone(settings.num_fewshot)
        self.assertIsNone(settings.limit)

    def test_accepts_chat_template_fewshot_and_limit(self) -> None:
        settings = BenchmarkSettings(
            task="mmlu_moral_scenarios",
            metric="acc,none",
            apply_chat_template=True,
            num_fewshot=0,
            limit=500,
        )

        self.assertTrue(settings.apply_chat_template)
        self.assertEqual(settings.limit, 500)


if __name__ == "__main__":
    unittest.main()
