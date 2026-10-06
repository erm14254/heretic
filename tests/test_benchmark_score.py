# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

from isolated_settings import IsolatedSettings

from heretic.scorers import benchmark_score
from heretic.scorers.benchmark_score import BenchmarkScore, Settings


class FakeWeights:
    """Stands in for the transformers model (a plain object() cannot be weakly referenced)."""


class StubHFLM:
    """Stores a pre-initialized model the way lm_eval's HFLM does."""

    def __init__(self, pretrained, tokenizer, batch_size):
        self._model = pretrained
        self.pretrained = pretrained
        self.tokenizer = tokenizer
        self.batch_size = batch_size


RESULTS = {"results": {"piqa": {"acc_norm,none": 0.75}}}


def make_scorer():
    model = SimpleNamespace(model=FakeWeights(), tokenizer=object())
    # The fork's Context exposes the model as `_model`.
    ctx = SimpleNamespace(_model=model)
    scorer = BenchmarkScore(
        heretic_settings=IsolatedSettings(model="tiny"),
        settings=Settings(),
    )
    with mock.patch.object(benchmark_score, "HFLM", StubHFLM):
        scorer.init(ctx)
    return scorer, ctx, model


class ModelReferenceTests(unittest.TestCase):
    def test_model_is_attached_only_while_evaluating(self) -> None:
        scorer, ctx, model = make_scorer()
        self.assertIsNone(scorer.hflm._model)
        self.assertIsNone(scorer.hflm.pretrained)

        attached = []

        def fake_evaluate(**kwargs):
            attached.append((kwargs["model"]._model, kwargs["model"].pretrained))
            return RESULTS

        with mock.patch.object(
            benchmark_score.lm_eval, "simple_evaluate", fake_evaluate
        ):
            score = scorer.get_score(ctx)

        self.assertEqual(attached, [(model.model, model.model)])
        self.assertEqual(score.value, 0.75)
        self.assertIsNone(scorer.hflm._model)
        self.assertIsNone(scorer.hflm.pretrained)

    def test_model_is_released_when_evaluation_fails(self) -> None:
        scorer, ctx, _ = make_scorer()

        with (
            mock.patch.object(
                benchmark_score.lm_eval,
                "simple_evaluate",
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaises(RuntimeError),
        ):
            scorer.get_score(ctx)

        self.assertIsNone(scorer.hflm._model)
        self.assertIsNone(scorer.hflm.pretrained)

    def test_released_model_can_be_garbage_collected(self) -> None:
        # Model.reset_model() drops its own reference before reloading the model
        # from disk; the scorer must not keep the old model (and its memory) alive.
        scorer, ctx, model = make_scorer()
        weights = weakref.ref(model.model)

        with mock.patch.object(
            benchmark_score.lm_eval, "simple_evaluate", return_value=RESULTS
        ):
            scorer.get_score(ctx)

        model.model = None
        gc.collect()

        self.assertIsNone(weights())


if __name__ == "__main__":
    unittest.main()
