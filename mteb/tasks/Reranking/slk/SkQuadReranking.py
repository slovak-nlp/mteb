from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class SkQuadReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SkQuadReranking",
        description=""" From Retrieval Sk QUAD """,
        reference="https://huggingface.co/datasets/TUKE-KEMT/reranking-skquad",
        dataset={
            "path": "TUKE-KEMT/reranking-skquad",
            "revision": "7cfcb122006b33d150b485caac5aa41939b3ba37",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map",
        date=("2025-10-09", "2025-10-09"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        bibtex_citation=r"""
@article{hladek2023slovak,
  author = {Hl{\'a}dek, Daniel and Sta{\v{s}}, J{\'a}n and Juh{\'a}r, Jozef and Koct{\'u}r, Tom{\'a}{\v{s}}},
  journal = {IEEE Access},
  pages = {32869--32881},
  publisher = {IEEE},
  title = {Slovak dataset for multilingual question answering},
  volume = {11},
  year = {2023},
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
    )
