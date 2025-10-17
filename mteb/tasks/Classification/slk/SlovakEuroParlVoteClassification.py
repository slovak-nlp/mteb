"""Slovak EuroParlVote classification tasks.

This module implements two classification tasks based on the EuroParlVote dataset,
using Slovak speeches from the European Parliament to predict:
- Gender of speakers (MALE/FEMALE)
- Vote position (FOR/AGAINST)

Note: Uses only speeches delivered in Slovak (Language=SK) with the full
speech text (Speech field).
"""

from __future__ import annotations

from typing import ClassVar

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{yang-etal-2025-demographics,
    title = "Demographics and Democracy: Benchmarking {LLM}s' Gender Bias and Political Leaning in {E}uropean Parliament",
    author = "Yang, Jinrui  and
      Han, Xudong  and
      Baldwin, Timothy",
    editor = "Abbas, Mourad  and
      Yousef, Tariq  and
      Galke, Lukas",
    booktitle = "Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP-2025)",
    month = aug,
    year = "2025",
    address = "Southern Denmark University, Odense, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.icnlsp-1.41/",
    pages = "416--439"
}
"""


class _EuroParlVoteSlovakMixin:
    """Shared transformation logic for EuroParlVote Slovak classification tasks.

    This mixin provides dataset transformation functionality for loading and processing
    the EuroParlVote dataset. It uses Slovak speeches (Language=SK) with full
    speech text (Speech field) for classification tasks.

    Attributes:
        target_column: The column name to use as the classification label.
    """

    target_column: ClassVar[str]

    def dataset_transform(self) -> None:
        """Transform the EuroParlVote dataset for classification.

        Performs the following steps:
        1. Filters for Slovak speeches (Language=SK) with valid text and labels
        2. Renames columns to standard MTEB format (text, label)
        3. Strips whitespace from text and labels
        4. Removes unnecessary columns
        5. Encodes labels as integers

        Note: The dataset already has train/validation/test splits on HuggingFace.
        We use train for training and test for evaluation, excluding validation.
        """
        target = self.target_column

        def _has_required_fields(example: dict) -> bool:
            """Check if example is Slovak with non-empty speech and label."""
            is_slovak = example.get("Language") == "SK"
            speech = example.get("Speech") or ""
            label = example.get(target) or ""
            return is_slovak and bool(speech.strip() and label.strip())

        # Filter each split for Slovak speeches only
        for split in self.dataset.keys():
            dataset = self.dataset[split]

            # Filter for Slovak speeches with valid content
            dataset = dataset.filter(_has_required_fields)

            # Rename to standard MTEB column names
            dataset = dataset.rename_columns({"Speech": "text", target: "label"})

            # Strip whitespace from text and labels
            dataset = dataset.map(
                lambda example: {
                    "text": example["text"].strip(),
                    "label": example["label"].strip(),
                }
            )

            # Remove all columns except text and label
            columns_to_remove = [
                column
                for column in dataset.column_names
                if column not in {"text", "label"}
            ]
            dataset = dataset.remove_columns(columns_to_remove)

            # Encode labels as integers
            dataset = dataset.class_encode_column("label")

            self.dataset[split] = dataset

        # Remove validation split (we only use train and test for MTEB evaluation)
        if "validation" in self.dataset:
            del self.dataset["validation"]


class SlovakEuroParlVoteGenderClassification(
    _EuroParlVoteSlovakMixin, AbsTaskClassification
):
    """Gender classification task using EuroParlVote Slovak speeches."""

    target_column: ClassVar[str] = "gender"

    metadata = TaskMetadata(
        name="SlovakEuroParlVoteGenderClassification",
        description="Binary classification task to predict the gender of Members of the European Parliament from Slovak speeches in the EuroParlVote dataset. Uses only speeches delivered in Slovak.",
        reference="https://arxiv.org/abs/2509.06164",
        dataset={
            "path": "unimelb-nlp/EuroParlVote",
            "revision": "48c50626aceb19b04da86ab32de47d2e8f4bb0c7",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-07-01", "2024-06-30"),  # 10th European Parliament term
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=["Political classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )


class SlovakEuroParlVotePositionClassification(
    _EuroParlVoteSlovakMixin, AbsTaskClassification
):
    """Vote position classification task using EuroParlVote Slovak speeches."""

    target_column: ClassVar[str] = "position"

    metadata = TaskMetadata(
        name="SlovakEuroParlVotePositionClassification",
        description="Binary classification task to predict the vote position (FOR/AGAINST) from Slovak speeches in the EuroParlVote dataset. Uses only speeches delivered in Slovak.",
        reference="https://arxiv.org/abs/2509.06164",
        dataset={
            "path": "unimelb-nlp/EuroParlVote",
            "revision": "48c50626aceb19b04da86ab32de47d2e8f4bb0c7",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-07-01", "2024-06-30"),  # 10th European Parliament term
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=["Political classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
    )
