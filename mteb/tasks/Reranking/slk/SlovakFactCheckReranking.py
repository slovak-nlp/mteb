from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


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
            "revision": "80be0018ee6781a4a7ced99649f7c79857883951",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map",
        date=("2020-01-01", "2025-03-31"),
        domains=["News", "Social", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{pikuliak-etal-2023-multilingual,
  address = {Singapore},
  author = {Pikuliak, Mat{\'u}{\v{s}}  and
Srba, Ivan  and
Moro, Robert  and
Hromadka, Timo  and
Smole{\v{n}}, Timotej  and
Meli{\v{s}}ek, Martin  and
Vykopal, Ivan  and
Simko, Jakub  and
Podrou{\v{z}}ek, Juraj  and
Bielikova, Maria},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2023.emnlp-main.1027},
  editor = {Bouamor, Houda  and Pino, Juan  and Bali, Kalika},
  month = dec,
  pages = {16477--16500},
  publisher = {Association for Computational Linguistics},
  title = {Multilingual Previously Fact-Checked Claim Retrieval},
  url = {https://aclanthology.org/2023.emnlp-main.1027},
  year = {2023},
}
""",
   prompt="Given a query, rerank the documents by their relevance to the query",
    )
