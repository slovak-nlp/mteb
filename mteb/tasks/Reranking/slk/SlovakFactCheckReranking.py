from __future__ import annotations

import datasets
from datasets import Dataset, DatasetDict

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking


_EVAL_SPLIT = "test"


class SlovakFactCheckReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SlovakFactCheckReranking",
        description=(
            "Created from Slovak part of the MultiClaim v2 dataset and curated for reranking task. "
            "The full dataset consists of 435k claims fact-checked by professional fact-checkers and "
            "89k social media posts containing these claims which were all published before April 2025. "
        ),
        reference="https://zenodo.org/records/15413169",
        dataset={
            "path": r"/home/user/workspace/slovak-nlp/mteb/slovak_factcheck_reranking_dummy.jsonl",
            "revision": "local",
            "trust_remote_code": True,
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["slk-Latn"],
        main_score="map",
        date=("2020-01-01", "2025-03-31"),
        domains=["News", "Social", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r""",
@inproceedings{pikuliak-etal-2023-multilingual,
  title = "Multilingual Previously Fact-Checked Claim Retrieval",
  author = "Pikuliak, Mat{\'u}{\v{s}} and Srba, Ivan and Moro, Robert and Hromadka, Timo and Smole{\v{n}}, Timotej and Meli{\v{s}}ek, Martin and Vykopal, Ivan and Simko, Jakub and Podrou{\v{z}}ek, Juraj and Bielikova, Maria",
  editor = "Bouamor, Houda  and Pino, Juan  and Bali, Kalika",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  month = dec,
  year = "2023",
  address = "Singapore",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.emnlp-main.1027",
  doi = "10.18653/v1/2023.emnlp-main.1027",
  pages = "16477--16500",
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_path = self.metadata.dataset["path"]

        if dataset_path.endswith(".jsonl"):
            print(f"🔍 Loading local JSONL dataset from {dataset_path}")
            dataset = Dataset.from_json(dataset_path)
        else:
            print(f"🔍 Loading dataset via Hugging Face Datasets hub: {dataset_path}")
            dataset = datasets.load_dataset(dataset_path, split=_EVAL_SPLIT)

        self.dataset = DatasetDict({"test": dataset})
        self.dataset_transform()
        self.data_loaded = True
