import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class WikiText2Dataset(Dataset):
    def __init__(self, split: str = "train", sequence_len: int = 128):
        self.block_size = sequence_len

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        all_text = "\n\n".join([example["text"] for example in dataset])

        tokenized = self.tokenizer(
            all_text,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
            truncation=False,
        )
        self.token_ids = tokenized["input_ids"]

        self.num_blocks = len(self.token_ids) // (sequence_len + 1)

        self.token_ids = self.token_ids[: self.num_blocks * (sequence_len + 1)]

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start_idx = idx * (self.block_size + 1)
        end_idx = start_idx + self.block_size + 1

        tokens = torch.tensor(self.token_ids[start_idx:end_idx], dtype=torch.long)

        inputs = tokens[:-1]
        targets = tokens[1:]

        return inputs, targets

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
