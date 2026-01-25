from typing import Iterator
import torch
import sentencepiece as spm
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset
import myneural as mn

SHUFFLE_BUFFER_SIZE = 1000

class PackedDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer: spm.SentencePieceProcessor, B, T) -> None:
        self.hf_dataset = hf_dataset
        self.context_length = T
        self.B = B
        self.sp = tokenizer
        self.eos_id = tokenizer.PieceToId('<EOS>')
        
    def tokenize_and_group(self, iterable, column="text"):
        iterator = iter(iterable)
        
        buffer = []
        block_size = self.B * self.context_length + 1
        for text_example in iterator:
            tokens = self.sp.EncodeAsIds(text_example[column])
            if len(tokens) == 0: continue
            tokens.append(self.eos_id)
            buffer.extend(tokens)
            
            while len(buffer) >= block_size:
                data= torch.tensor(buffer[:block_size], dtype=torch.long)
                xs = data[:-1].view(self.B, self.context_length) # (B, T)
                ys = data[1:].view(self.B, self.context_length) # (B, T)
                yield xs, ys
                buffer=buffer[block_size-1:]
                
    def __iter__(self) -> Iterator:
        return self.tokenize_and_group(self.hf_dataset)
    
def get_interleaved_dataset():
    smol_data_sources = [
        "cosmopedia-v2",
        "fineweb-edu-dedup",
    ]
    
    hf_datasources = []
    for source in smol_data_sources:
        ds = load_dataset("HuggingFaceTB/smollm-corpus", source, split='train', streaming=True)
        hf_datasources.append(ds)
        
    ds = load_dataset("transformersbook/codeparrot", split='train',streaming=True)
    ds=ds.rename_column('content', 'text')
    hf_datasources.append(ds)        
    
    probs = [0.75, 0.15, 0.1] # cosmopedia, fineweb-edu, python 
    mixed_dataset = interleave_datasets(hf_datasources, probs, seed=42, stopping_strategy="all_exhausted")
    
    return mixed_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=42)

def train_tokenizer(input_path, vocab_size = 32768, max_samples = 250_000):
            
    sp = spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix = 'models/my_tokenizer',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage = 0.9995,
        input_sentence_size = max_samples,
        shuffle_input_sentence = True,
        byte_fallback = True,
        split_digits = True,
        allow_whitespace_only_pieces = True,
        remove_extra_whitespaces = False,
        user_defined_symbols = ['<start_of_turn>', '<end_of_turn>', '<EOS>'],
        pad_id = 0,
        unk_id = 1,
        bos_id = 2,
        eos_id = 3, # won't really be using this but keeping it in for safety
    )
    
def create_tokenizer_corpus(dataset, max_samples = 250_000):
    with open('data/token-corpus.txt', 'w', encoding='utf-8') as f:
        for idx, text_example in enumerate(dataset):
            text = text_example["text"]

            if text is not None:
                f.write(text)
            else: print(f"Text was none at {idx}; most likely one or more of the interleaved datasets don't have 'text' column")
            if idx >= max_samples:
                break
    return 'data/token-corpus.txt'