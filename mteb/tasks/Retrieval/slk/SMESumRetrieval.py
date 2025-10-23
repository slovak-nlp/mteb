from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SMESumRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SMESumRetrieval",
        description="""
            SMESum, a Slovak news summarization dataset consisting of 80,000
            news articles with titles and introductions obtained from the SME news portal.

            Originally intended as a summarization task, but since no human annotations were provided
            here reformulated to a retrieval task. The task uses article introductions as queries
            to retrieve full documents, representing a realistic use case of finding articles
            from their abstracts.
        """,
        reference="https://huggingface.co/datasets/NaiveNeuron/SMESum",
        dataset={
            "path": "NaiveNeuron/SMESum",
            "revision": "c5a6521a4ddce3450fb04ba218623681a9189c6d",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="ndcg_at_10",
        date=("2013-01-01", "2019-12-31"),
        domains=["News", "Social", "Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{suppa-adamec-2020-summarization,
    title = "A Summarization Dataset of {S}lovak News Articles",
    author = "Suppa, Marek and Adamec, Jergus",
    editor = "Calzolari, Nicoletta and B{\'e}chet, Fr{\'e}d{\'e}ric and Blache, Philippe and Choukri, Khalid and Cieri, Christopher and Declerck, Thierry and Goggi, Sara and Isahara, Hitoshi and Maegaard, Bente and Mariani, Joseph and Mazo, H{\'e}l{\`e}ne and Moreno, Asuncion and Odijk, Jan and Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.830/",
    pages = "6725--6730",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        n_sample = 600

        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            split_ds = datasets.load_dataset(
                dataset_path, split=f"{split}[:{n_sample}]"
            )
            # Use introduction only as query to avoid title overlap with corpus
            queries = {f"q{e + 1}": x["introduction"] for e, x in enumerate(split_ds)}
            corpus = {
                f"d{e + 1}": {"title": x["title"], "text": x["document"]}
                for e, x in enumerate(split_ds)
            }
            qrels = {f"q{i + 1}": {f"d{i + 1}": 1} for i in range(split_ds.shape[0])}
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )
        self.data_loaded = True
