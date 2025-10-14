from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SlovakParlaSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SlovakParlaSentClassification",
        description="Slovak parliamentary sentiment classification dataset from the ParlaSent corpus. Contains sentences from parliamentary debates with 3-level sentiment annotations.",
        reference="https://huggingface.co/datasets/classla/ParlaSent",
        dataset={
            "path": "classla/ParlaSent",
            "name": "SK",
            "revision": "0587c2b6499fbc68a7623439c2af2b24748968dc",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2018-01-01", "2018-12-31"),
        eval_splits=["train"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{antunovic2022parlasent,
  title={ParlaSent: A multilingual sentiment analysis dataset of parliamentary debates},
  author={Antunovi{\'c}, Matej and Bra{\v{z}}inskas, Rytis and {\v{Z}}agar, Bojan and Haddow, Barry and Birch, Alexandra and Ljube{\v{s}}i{\'c}, Nikola},
  journal={arXiv preprint arXiv:2210.03068},
  year={2022}
}
""",
    )

    def dataset_transform(self) -> None:
        # Rename 'sentence' column to 'text' as expected by MTEB
        self.dataset = self.dataset.rename_columns({"sentence": "text"})
