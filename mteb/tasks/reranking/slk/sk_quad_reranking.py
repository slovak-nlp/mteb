from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SkQuadReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SkQuadReranking",
        description=""" From Retrieval Sk QUAD """,
        reference="https://huggingface.co/datasets/TUKE-KEMT/reranking-skquad",
        dataset={
            "path": "TUKE-KEMT/reranking-skquad",
            "revision": "3997eba1c60721e34f7b01db400f5b14a40218ea",
        },
        type="Reranking",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map_at_1000",
        date=("2025-10-09", "2025-10-09"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        bibtex_citation=r"""
""",
    )
