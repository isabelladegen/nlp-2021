from datasets import load_dataset
import pandas as pd


def load_rc_dataset(split: str):
    rc_dataset = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split=split,
        ignore_verifications=True,
        cache_dir="./data_cache_src"
    )
    return rc_dataset


class RCData:
    def __init__(self, split: str):
        self.rc_dataset = load_rc_dataset(split)

    def raw_questions_and_gold_answers(self) -> pd.DataFrame:
        rows = []
        for example in self.rc_dataset:
            rows.append([example['id'], example['title'], example['question'], example['answers']])
        return pd.DataFrame(rows, columns=["id", "doc_id", "question", "answers"])
