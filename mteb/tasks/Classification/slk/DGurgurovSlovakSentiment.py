from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DGurgurovSlovakSentiment(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DGurgurovSlovakSentiment",
        description="Sentiment Analysis Data for the Slovak Language. Binary sentiment classification dataset with ~5,000 samples used for improving word embeddings with graph knowledge for low resource languages.",
        reference="https://aclanthology.org/W19-3716/",
        dataset={
            "path": "DGurgurov/slovak_sa",
            "revision": "250a73199a3013bf9bf6b73b3fbdf83279b40375",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2019-01-01", "2019-08-01"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{pecar-etal-2019-improving,
  address = {Florence, Italy},
  author = {Pecar, Samuel and Simko, Marian and Bielikova, Maria},
  booktitle = {Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing},
  doi = {10.18653/v1/W19-3716},
  month = aug,
  pages = {114--119},
  publisher = {Association for Computational Linguistics},
  title = {Improving Sentiment Classification in {S}lovak Language},
  year = {2019},
}
""",
    )

    def dataset_transform(self) -> None:
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
