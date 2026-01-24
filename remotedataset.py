from typing import Iterator
import torch
import sentencepiece as spm
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset
import myneural as mn
class PackedDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer: spm.SentencePieceProcessor, context_length) -> None:
        self.hf_dataset = hf_dataset
        self.context_length = context_length
        self.sp = tokenizer
        
    def tokenize_and_group(self, iterable):
        iterator = iter(iterable)
        
        buffer = []
        
        for text_example in iterator:
            tokens = self.sp.EncodeAsIds(text_example["text"])
            buffer.extend(tokens)
            
            while len(buffer) >= self.context_length:
                yield torch.tensor(buffer[:self.context_length], dtype=torch.long)
                buffer=buffer[self.context_length:]
                
    def __iter__(self) -> Iterator:
        return self.tokenize_and_group(self.hf_dataset)
    
def get_interleaved_dataset():
    data_sources = [
        "cosmopedia-v2",
        "fineweb-edu-dedup",
        "python-edu"
    ]
    
    probs = [0.8, 0.1, 0.1]
    hf_datasources = []
    for source in data_sources:
        ds = load_dataset("HuggingFaceTB/smollm-corpus", source, split='train', streaming=True)
        hf_datasources.append(ds)
        
    mixed_dataset = interleave_datasets(hf_datasources, probs, seed=42, stopping_strategy="all_exhausted")
    
    return mixed_dataset.shuffle(batch_size=10000, seed=42)