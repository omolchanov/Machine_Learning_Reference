from datasets import load_dataset
from env_config import *


class LlmCorpus:

    @staticmethod
    def _load_external_ds(id, group_id=None, trust_remote_code=False):
        split = f'train[:{DATASET_SIZE}]'
        if group_id:
            ds = load_dataset(id, group_id, split=split, trust_remote_code=trust_remote_code)
        else:
            ds = load_dataset(id, split=split, trust_remote_code=trust_remote_code)
        print(f"{id} {group_id or ''} dataset loaded")
        return ds

    @staticmethod
    def get_corpus():
        """
        Loads 3 datasets from Hugging faces and combines them into a single corpus
        """
        books = LlmCorpus._load_external_ds('bookcorpus', trust_remote_code=True)
        wiki2 = LlmCorpus._load_external_ds('wikitext', 'wikitext-2-raw-v1')
        wiki103 = LlmCorpus._load_external_ds('wikitext', 'wikitext-103-raw-v1')

        corpus = "\n".join([
            "\n".join(books["text"]).lower(),
            "\n".join(wiki2["text"]).lower(),
            "\n".join(wiki103["text"]).lower(),
        ])

        return corpus
