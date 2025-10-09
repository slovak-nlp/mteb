from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SlovakNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SlovakNLI",
        description="Slovak Handwritten Annotated NLI dataset",
        reference="https://huggingface.co/datasets/natalia-nk/NLI-SK-annotated",
        dataset={
            "path": "natalia-nk/NLI-SK-annotated",
            "revision": "220b9cfa730c36ec563517fcd749c9c7a5ae4436",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="max_ap",
        annotations_creators="human-annotated",
    )

    def dataset_transform(self):
        _dataset = {}

        # TODO: The dataset currently has only 'train' split
        hf_dataset = self.dataset["train"].filter(
            lambda x: x["Label"] in ["Entailment", "Contradiction"])
        hf_dataset = hf_dataset.map(
            lambda example: {"Label": 1 if example["Label"] == "Entailment" else 0}
        )

        _dataset["test"] = [
            {
                "sentence1": hf_dataset["Premise"],
                "sentence2": hf_dataset["Hypothesis"],
                "labels": hf_dataset["Label"],
            }
        ]

        self.dataset = _dataset
