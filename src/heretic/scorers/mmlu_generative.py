# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Generative MMLU capability guard.

The model is asked a fixed set of multiple-choice questions in chat form and a
lenient parser reads the answer letter from what it writes, the way an
external evaluation harness does. This differs from `BenchmarkScore` with an
MMLU task, which compares the likelihoods of the bare answer letters as
continuations: on models that do not naturally start a reply with a bare
letter (e.g. Qwen in no-think mode) that likelihood ranking is dominated by
formatting preferences, scores far below the model's real accuracy, and can
move by ten points for a change in style. Generating and parsing measures what
the model actually answers.

Unparsed replies count as wrong, like a harness would count them, and their
number is reported alongside the accuracy.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, PositiveInt
from rich.markup import escape

from heretic.scorer import Context, Score, Scorer
from heretic.utils import Prompt, batchify, print

MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)

LETTERS = "ABCD"

DEFAULT_INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "Reply with the letter of the correct option only."
)


class Settings(BaseModel):
    score_name: str = Field(
        default="MMLU (generative)",
        description="Name that describes what the configured score measures.",
    )

    per_subject: PositiveInt = Field(
        default=20,
        description=(
            "Number of questions per subject, taken from the start of the test "
            "split in dataset order, so every trial sees the same questions."
        ),
    )

    subjects: list[str] = Field(
        default_factory=lambda: list(MMLU_SUBJECTS),
        description="MMLU subjects to include (default: all 57).",
    )

    instruction: str = Field(
        default=DEFAULT_INSTRUCTION,
        description="Text placed before the question in the user message.",
    )

    max_new_tokens: PositiveInt = Field(
        default=16,
        description=(
            "Generation budget per question. A letter needs one or two tokens; "
            "the rest absorbs preambles such as 'The answer is'."
        ),
    )

    print_failures: bool = Field(
        default=False,
        description="Whether to print the replies the parser could not read.",
    )

    dataset: str = Field(
        default="cais/mmlu",
        description="Hugging Face dataset with MMLU subjects as configurations.",
    )

    split: str = Field(
        default="test", description="Dataset split to draw questions from."
    )


@dataclass
class Question:
    subject: str
    text: str
    choices: list[str]
    answer: int  # Index into choices.


def format_question(question: Question, instruction: str) -> str:
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, question.choices)
    )
    return f"{instruction}\n\n{question.text.strip()}\n\n{options}"


# Patterns tried in order on the cleaned reply. Each captures the letter.
ANSWER_PATTERNS = (
    # "The answer is B", "Answer: C", "correct answer: (D)"
    re.compile(r"answer\s*(?:is|:)?\s*\(?([ABCD])(?![A-Za-z])", re.IGNORECASE),
    # "Option B", "choice C"
    re.compile(r"(?:option|choice)\s*\(?([ABCD])(?![A-Za-z])", re.IGNORECASE),
    # A reply that starts with the letter: "B", "B.", "(B)", "B) ...", "B: Paris"
    re.compile(r"^\s*\(?([ABCD])(?:[.:)\]]|\s|$)"),
    # The same in lower case, but only as a bare letter or with punctuation,
    # so that a sentence starting with the article "a" does not count.
    re.compile(r"^\s*\(?([abcd])(?:[.:)\]]|$)"),
)
# Fallback: a lone upper-case letter anywhere in the first line.
LONE_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def parse_answer(response: str) -> str | None:
    """
    Return the answer letter a reply expresses, or None if none can be read.
    """
    cleaned = re.sub(r"[*_`#]", "", response).strip()
    if not cleaned:
        return None

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return match.group(1).upper()

    first_line = cleaned.splitlines()[0]
    match = LONE_LETTER.search(first_line)
    if match:
        return match.group(1)

    return None


def load_questions(
    dataset: str, split: str, subjects: list[str], per_subject: int
) -> list[Question]:
    # Imported here so the module stays importable without the datasets package.
    from datasets import load_dataset

    questions: list[Question] = []
    for subject in subjects:
        rows = load_dataset(dataset, subject, split=split)
        for row in rows.select(range(min(per_subject, len(rows)))):
            questions.append(
                Question(
                    subject=subject,
                    text=row["question"],
                    choices=list(row["choices"]),
                    answer=int(row["answer"]),
                )
            )
    return questions


class MMLUGenerative(Scorer):
    """
    Accuracy on a fixed MMLU panel, answered generatively and parsed leniently.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading MMLU questions from [bold]{self.settings.dataset}[/] "
            f"({self.settings.per_subject} per subject, {len(self.settings.subjects)} subjects)..."
        )
        self.questions = load_questions(
            self.settings.dataset,
            self.settings.split,
            self.settings.subjects,
            self.settings.per_subject,
        )
        print(f"* [bold]{len(self.questions)}[/] questions loaded")

    def generate_answers(self, model, prompts: list[Prompt]) -> list[str]:
        """
        Short greedy generations for the prompts, prompt part stripped.
        Works with both backends: each exposes generate() returning the
        tokenized inputs and the prompt-inclusive output ids, plus a tokenizer.
        """
        responses: list[str] = []
        batch_size = max(1, self.heretic_settings.batch_size)
        for batch in batchify(prompts, batch_size):
            inputs, outputs = model.generate(
                batch, max_new_tokens=self.settings.max_new_tokens
            )
            prompt_length = inputs["input_ids"].shape[1]
            responses.extend(
                model.tokenizer.batch_decode(
                    outputs[:, prompt_length:], skip_special_tokens=True
                )
            )
        return responses

    def get_score(self, ctx: Context) -> Score:
        prompts = [
            Prompt(
                system=self.heretic_settings.system_prompt,
                user=format_question(question, self.settings.instruction),
            )
            for question in self.questions
        ]
        responses = self.generate_answers(ctx._model, prompts)

        correct = 0
        unparsed = 0
        for question, response in zip(self.questions, responses):
            letter = parse_answer(response)
            if letter is None:
                unparsed += 1
                if self.settings.print_failures:
                    print(f"[yellow]Unparsed reply:[/] {escape(response.strip())}")
            elif letter == LETTERS[question.answer]:
                correct += 1

        total = len(self.questions)
        accuracy = correct / total if total else 0.0

        return Score(
            value=accuracy,
            rich_display=f"[bold]{accuracy:.4f}[/] ({correct}/{total}, {unparsed} unparsed)",
            md_display=f"{accuracy:.4f} ({correct}/{total}, {unparsed} unparsed)",
        )
