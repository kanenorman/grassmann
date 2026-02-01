import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class WikiText2Dataset(Dataset):
    def __init__(self, split: str = "train", block_size: int = 128):
        self.block_size = block_size

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        all_text = "\n\n".join([example["text"] for example in dataset])

        tokenized = self.tokenizer(
            all_text,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        self.token_ids = tokenized["input_ids"]

        self.num_blocks = len(self.token_ids) // block_size

        self.token_ids = self.token_ids[: self.num_blocks * block_size]

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size

        tokens = torch.tensor(self.token_ids[start_idx:end_idx], dtype=torch.long)

        inputs = tokens[:-1]
        targets = tokens[1:]

        return inputs, targets

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
