# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import lm_eval
from lm_eval.models.huggingface import HFLM
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

from heretic.scorer import Context, Score, Scorer


class Settings(BaseModel):
    score_name: str = Field(
        default="PIQA acc_norm",
        description="Name that describes what the configured benchmark score measures.",
    )

    task: str = Field(
        default="piqa",
        description="Task ID of the benchmark in the Language Model Evaluation Harness.",
    )

    metric: str = Field(
        default="acc_norm,none",
        description="Task metric to use as the benchmark score.",
    )

    apply_chat_template: bool = Field(
        default=False,
        description=(
            "Whether to wrap each benchmark prompt in the model's chat template. "
            "Instruction-tuned models often score near chance on the harness's raw "
            "completion format and like they do in chat with the template applied."
        ),
    )

    num_fewshot: NonNegativeInt | None = Field(
        default=None,
        description="Number of few-shot examples to include (unset = the task's default).",
    )

    limit: PositiveInt | None = Field(
        default=None,
        description="Evaluate only the first N examples of the task (unset = all).",
    )


class BenchmarkScore(Scorer):
    """
    Calculates the score of a benchmark from the Language Model Evaluation Harness.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        self.hflm = HFLM(
            pretrained=ctx._model.model,  # ty:ignore[invalid-argument-type]
            tokenizer=ctx._model.tokenizer,  # ty:ignore[invalid-argument-type]
            batch_size="auto",
        )

    def get_score(self, ctx: Context) -> Score:
        # The purpose of this hack, where we initialize the HFLM object once,
        # then update its internal model every time we calculate the score,
        # is to get the benefits of batch size caching while allowing for
        # model reloads, e.g. when using --evaluate-model.
        self.hflm.pretrained = ctx._model.model
        self.hflm._model = ctx._model.model

        results = lm_eval.simple_evaluate(
            model=self.hflm,
            tasks=[self.settings.task],
            num_fewshot=self.settings.num_fewshot,
            limit=self.settings.limit,
            apply_chat_template=self.settings.apply_chat_template,
            fewshot_as_multiturn=(
                self.settings.apply_chat_template
                and (self.settings.num_fewshot or 0) > 0
            ),
        )

        benchmark_score = float(
            results["results"][self.settings.task][self.settings.metric]
        )

        return Score(
            value=benchmark_score,
            rich_display=f"[bold]{benchmark_score:.4f}[/]",
            md_display=f"{benchmark_score:.4f}",
        )
