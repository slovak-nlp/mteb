from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class AskUbuntuDupQuestionsVN(AbsTaskReranking):
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
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        bibtex_citation=r"""
""",
        adapted_from=["SlovakRetrieval"],
    )
