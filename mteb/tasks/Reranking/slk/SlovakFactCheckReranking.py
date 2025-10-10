from __future__ import annotations

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
            "path": "kinit/sk-factcheck-reranking",
            "revision": "0ee24363ae8328448093d0d6784cbf0fde499469",
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
        license="cc-by-nc-nd-4.0",
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
