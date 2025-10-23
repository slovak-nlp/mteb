from __future__ import annotations

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class SlovakSumURLClustering(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="SlovakSumURLClustering",
        description="Clustering of Slovak news articles from SlovakSum dataset based on the URL structure. Articles are organized into 12 editorial categories including sports, culture, economy, health, travel, politics, and technology sections.",
        reference="https://aclanthology.org/2024.lrec-main.1298/",
        dataset={
            "path": "kiviki/slovaksum-url-clustering",
            "revision": "ac3b8aafe34e9ee47f5db377aec524dfcdf885c5",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2015-04-26", "2022-01-11"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ondrejova-suppa-2024-slovaksum,
  address = {Torino, Italia},
  author = {Ondrejova, Viktoria and Suppa, Marek},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  editor = {Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen},
  month = may,
  pages = {14916--14922},
  publisher = {ELRA and ICCL},
  title = {{S}lovak{S}um: A Large Scale {S}lovak Summarization Dataset},
  url = {https://aclanthology.org/2024.lrec-main.1298/},
  year = {2024},
}
""",
        prompt="Identify the topic or theme of the given text.",
    )

    def dataset_transform(self):
        """Transform the dataset to create sentences (title + summary) and labels (url_category)."""
        ds = {}
        for split in self.metadata.eval_splits:
            # Combine title and summary to create sentences
            titles = self.dataset[split]["title"]
            summaries = self.dataset[split]["sum"]

            sentences = [
                f"{title} {summary}".strip()
                for title, summary in zip(titles, summaries)
            ]

            labels = self.dataset[split]["theme"]

            ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        self.dataset = DatasetDict(ds)
