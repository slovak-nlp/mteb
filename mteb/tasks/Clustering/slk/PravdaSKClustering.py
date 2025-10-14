from __future__ import annotations

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata

_MAX_DOCUMENT_TO_EMBED = 2048


class PravdaSKTagClustering(AbsTaskClusteringFast):
    max_document_to_embed = _MAX_DOCUMENT_TO_EMBED
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="PravdaSKTagClustering",
        description="Clustering of Slovak news articles from Pravda.sk based on article tags. Articles are grouped into 50 thematic categories including Slovak politics, international affairs, events, and topics.",
        reference="https://huggingface.co/datasets/NaiveNeuron/pravda-sk-tag-clustering",
        dataset={
            "path": "NaiveNeuron/pravda-sk-tag-clustering",
            "revision": "dd0a6c077151b8c8bc2fd6abcd746b34fde80bf8",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2014-01-01", "2024-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Identify topic categories based on article tags in Slovak news",
    )

    def dataset_transform(self):
        """Transform the dataset to create sentences (title + summary) and labels (assigned_label)."""
        ds = {}
        for split in self.metadata.eval_splits:
            # Combine title and summary to create sentences
            titles = self.dataset[split]["title"]
            summaries = self.dataset[split]["summary"]

            sentences = [
                f"{title} {summary}".strip()
                for title, summary in zip(titles, summaries)
            ]

            labels = self.dataset[split]["assigned_label"]

            ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        self.dataset = DatasetDict(ds)


class PravdaSKURLClustering(AbsTaskClusteringFast):
    max_document_to_embed = _MAX_DOCUMENT_TO_EMBED
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="PravdaSKURLClustering",
        description="Clustering of Slovak news articles from Pravda.sk based on URL structure. Articles are organized into 50 editorial categories reflecting pravda.sk's content organization, including news, sports, culture, economy, health, travel, celebrity, and science sections.",
        reference="https://huggingface.co/datasets/NaiveNeuron/pravda-sk-url-clustering",
        dataset={
            "path": "NaiveNeuron/pravda-sk-url-clustering",
            "revision": "90acfb72391c5952e9da8233d42fbbc49182cd20",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="v_measure",
        date=("2014-01-01", "2024-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Identify editorial categories based on URL structure in Slovak news",
    )

    def dataset_transform(self):
        """Transform the dataset to create sentences (title + summary) and labels (url_category)."""
        ds = {}
        for split in self.metadata.eval_splits:
            # Combine title and summary to create sentences
            titles = self.dataset[split]["title"]
            summaries = self.dataset[split]["summary"]

            sentences = [
                f"{title} {summary}".strip()
                for title, summary in zip(titles, summaries)
            ]

            labels = self.dataset[split]["url_category"]

            ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        self.dataset = DatasetDict(ds)
