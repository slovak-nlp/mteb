from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SlovakPharmacyDrMaxReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SlovakPharmacyDrMaxReranking",
        description=(
            "A reranking dataset created from Q&A content collected from DrMax pharmacy website. "
            "The dataset consists of questions about medications, health conditions, and pharmaceutical advice, "
            "with answers provided by qualified pharmacists. This dataset is designed to evaluate models' "
            "ability to rank relevant pharmaceutical information and expert responses."
        ),
        reference="https://huggingface.co/datasets/kinit/slovak-pharmacy-drmax-reranking",
        dataset={
            "path": "slovak-nlp/slovak-pharmacy-drmax-reranking",
            "revision": "faf73c5217c37e8a9ba297b8e29b25d1ccdeccae",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map_at_1000",
        date=("2025-11-01", "2025-11-30"),
        domains=[
            "Medical",
            "Web",
        ],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""""",
    )


class SlovakPharmacyMojaLekarenReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SlovakPharmacyMojaLekarenReranking",
        description=(
            "A reranking dataset created from Q&A content collected from MojaLekaren pharmacy website. "
            "The dataset consists of questions about medications, health conditions, and pharmaceutical advice, "
            "with answers provided by qualified pharmacists. This dataset is designed to evaluate models' "
            "ability to rank relevant pharmaceutical information and expert responses."
        ),
        reference="https://huggingface.co/datasets/slovak-nlp/slovak-pharmacy-mojalekaren-reranking",
        dataset={
            "path": "slovak-nlp/slovak-pharmacy-mojalekaren-reranking",
            "revision": "962818770efbb0395130a3abb04dcae3b4ef7c30",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map_at_1000",
        date=("2025-11-01", "2025-11-30"),
        domains=[
            "Medical",
            "Web",
        ],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""""",
    )
