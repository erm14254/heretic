# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace

from isolated_settings import IsolatedSettings

from heretic.scorers.mmlu_generative import (
    MMLU_SUBJECTS,
    MMLUGenerative,
    Question,
    Settings,
    format_question,
    parse_answer,
)


class ParseAnswerTests(unittest.TestCase):
    def test_reads_the_common_reply_shapes(self) -> None:
        cases = {
            "B": "B",
            "b": "B",
            "B.": "B",
            "(C)": "C",
            "D) Paris": "D",
            "A: because it is the capital": "A",
            "**C**": "C",
            "The answer is B.": "B",
            "The correct answer is (D).": "D",
            "Answer: C": "C",
            "I would pick option B here.": "B",
            "Choice A": "A",
            "Looking at this, A": "A",
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(parse_answer(reply), expected)

    def test_returns_none_when_no_letter_can_be_read(self) -> None:
        for reply in ("", "   ", "I don't know.", "The capital is Paris.", "abcd"):
            with self.subTest(reply=reply):
                self.assertIsNone(parse_answer(reply))

    def test_prefers_an_explicit_answer_statement_over_a_leading_article(self) -> None:
        self.assertEqual(parse_answer("A good look shows the answer is C."), "C")


class FormatQuestionTests(unittest.TestCase):
    def test_renders_instruction_question_and_lettered_options(self) -> None:
        question = Question(
            "astronomy",
            "  Which planet is largest?  ",
            ["Mars", "Jupiter", "Venus", "Earth"],
            1,
        )

        text = format_question(question, "Reply with the letter only.")

        self.assertEqual(
            text,
            "Reply with the letter only.\n\nWhich planet is largest?\n\nA. Mars\nB. Jupiter\nC. Venus\nD. Earth",
        )


class ScoringTests(unittest.TestCase):
    def make_scorer(self, replies: list[str]) -> tuple[MMLUGenerative, list]:
        scorer = MMLUGenerative(
            heretic_settings=IsolatedSettings(model="tiny"),
            settings=Settings(per_subject=2, subjects=["astronomy"]),
        )
        scorer.questions = [
            Question("astronomy", "Q1", ["a", "b", "c", "d"], 1),  # B
            Question("astronomy", "Q2", ["a", "b", "c", "d"], 3),  # D
            Question("astronomy", "Q3", ["a", "b", "c", "d"], 0),  # A
            Question("astronomy", "Q4", ["a", "b", "c", "d"], 2),  # C
        ]
        seen: list = []

        def fake_generate_answers(model, prompts):
            seen.extend(prompts)
            return replies

        scorer.generate_answers = fake_generate_answers  # type: ignore[method-assign]
        return scorer, seen

    def test_counts_correct_wrong_and_unparsed_replies(self) -> None:
        scorer, seen = self.make_scorer(["B", "The answer is A.", "no idea", "**C**"])

        score = scorer.get_score(SimpleNamespace(_model=object()))

        # Correct: Q1 (B), Q4 (C). Wrong: Q2 (said A, is D). Unparsed: Q3.
        self.assertAlmostEqual(score.value, 0.5)
        self.assertIn("2/4", score.rich_display)
        self.assertIn("1 unparsed", score.rich_display)
        self.assertEqual(len(seen), 4)
        self.assertTrue(seen[0].user.endswith("A. a\nB. b\nC. c\nD. d"))

    def test_default_settings_cover_all_subjects(self) -> None:
        settings = Settings()

        self.assertEqual(len(settings.subjects), 57)
        self.assertEqual(settings.subjects, list(MMLU_SUBJECTS))
        self.assertEqual(settings.per_subject, 20)
        self.assertEqual(settings.max_new_tokens, 16)


if __name__ == "__main__":
    unittest.main()
