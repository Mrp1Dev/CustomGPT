from typing import Iterator
import torch
import sentencepiece as spm
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset
import re
import json
SHUFFLE_BUFFER_SIZE = 10000
class PackedDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer: spm.SentencePieceProcessor, B, T, examples_to_skip=0) -> None:
        self.hf_dataset = hf_dataset
        self.context_length = T
        self.B = B
        self.sp = tokenizer
        self.eos_id = tokenizer.PieceToId('<EOS>')
        self.examples_to_skip = examples_to_skip
        self.nl_id = tokenizer.PieceToId('<0x0A>')
        self.tab_id = tokenizer.PieceToId('<0x09>')
        self.r_id = tokenizer.PieceToId('<0x0D>')
        self.split_pattern = re.compile(r'(\n|\t|\r)')
        self.user_id = [tokenizer.PieceToId('<start_of_turn>'), tokenizer.PieceToId('user')]
        self.model_id = [tokenizer.PieceToId('<start_of_turn>'), tokenizer.PieceToId('model')]
        self.system_id = [tokenizer.PieceToId('<start_of_turn>'), tokenizer.PieceToId('system')]
        self.end_of_turn_id = [tokenizer.PieceToId('<end_of_turn>')]
        
    def tokenize_and_group(self, iterable, examples_to_skip=0, column="messages"):
        while True:
            iterator = iter(iterable)
            
            xs_buffer = []
            ys_buffer = []
            block_size = self.B * self.context_length + 1
            if examples_to_skip > 0:
                print(f"Worker skipping {examples_to_skip} out of {self.examples_to_skip} (total) examples.")
            for i, text_example in enumerate(iterator):
                if i < examples_to_skip:
                    if i == self.examples_to_skip - 1:
                        print(f"Worker successfully skipped {examples_to_skip} examples.")
                    continue
                
                xs_ids, ys_ids = self.encode_json(text_example[column]) # shape xs_ids == ys_ids, we'll leave the last one for xs and leave the first one for ys later.
                if len(xs_ids) == 0: continue
                xs_ids.append(self.eos_id)
                ys_ids.append(self.eos_id)
                
                xs_buffer.extend(xs_ids)
                ys_buffer.extend(ys_ids)
                while len(xs_buffer) > block_size:
                    end = block_size #min(block_size, len(xs_buffer))
                    current_ys = ys_buffer[:end]

                    if max(current_ys) == -100:
                        xs_buffer=xs_buffer[end-1:]
                        ys_buffer=ys_buffer[end-1:]
                        continue
                    
                    xs_tensor= torch.tensor(xs_buffer[:end], dtype=torch.long)
                    ys_tensor= torch.tensor(current_ys, dtype=torch.long)
                    #----No padding, i'm packing instead.------
                    # if xs_tensor.shape[0] < block_size:
                    #     padding_required = block_size-xs_tensor.shape[0]
                    #     xs_tensor = torch.nn.functional.pad(xs_tensor, (0, padding_required), value=self.sp.pad_id())
                    #     ys_tensor = torch.nn.functional.pad(ys_tensor, (0, padding_required), value=-100)
                    
                    xs = xs_tensor[:-1].view(self.B, self.context_length) # (B, T)
                    ys = ys_tensor[1:].view(self.B, self.context_length) # (B, T)
                    yield xs, ys
                    xs_buffer=xs_buffer[end-1:]
                    ys_buffer=ys_buffer[end-1:]
            examples_to_skip=0
            print("--------EPOCH OVER, LOOPING.--------")        
    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.tokenize_and_group(self.hf_dataset, self.examples_to_skip)
        else:
            sharded_dataset = self.hf_dataset.shard(num_shards = worker_info.num_workers, index=worker_info.id)
            examples_to_skip = self.examples_to_skip // worker_info.num_workers
            return self.tokenize_and_group(sharded_dataset, examples_to_skip)
    
    def encode_json(self, json_text):
        text = json.loads(json_text) if isinstance(json_text, str) else json_text
        
        xs_ids = []
        ys_ids = []
        for message in text:
            role = message['role']
            if role == 'user':
                part_xs = self.user_id + self.encode_with_whitespace(message['content']) + self.end_of_turn_id
                part_ys = [-100] * len(part_xs)
            elif role == 'assistant':
                content_ids = self.encode_with_whitespace(message['content']) + self.end_of_turn_id
                part_xs = self.model_id + content_ids
                part_ys = [-100] * len(self.model_id) + content_ids
            else:
                part_xs = self.system_id + self.encode_with_whitespace(message['content']) + self.end_of_turn_id # some possibly system prompts etc
                part_ys = [-100] * len(part_xs)
                
            xs_ids.extend(part_xs)
            ys_ids.extend(part_ys)    
        return xs_ids, ys_ids
        
    def encode_with_whitespace(self, text):
        splits = self.split_pattern.split(text)
        ids = []
        for split in splits:
            if split == '\n':
                ids.append(self.nl_id)
            elif split == '\t':
                ids.append(self.tab_id)
            elif split == '\r':
                ids.append(self.r_id)
            else:
                ids.extend(self.sp.EncodeAsIds(split))
        return ids
                    
    
def get_interleaved_dataset():
    smol_data_sources = [
        "cosmopedia-v2",
        "fineweb-edu-dedup",
    ]
    
    hf_datasources = []
    for source in smol_data_sources:
        ds = load_dataset("HuggingFaceTB/smollm-corpus", source, split='train', streaming=True)#, token=hf_token)
        hf_datasources.append(ds)
        
    ds = load_dataset("transformersbook/codeparrot", split='train',streaming=True)#, token=hf_token)
    ds=ds.rename_column('content', 'text')
    hf_datasources.append(ds)        
    
    probs = [0.75, 0.15, 0.1] # cosmopedia, fineweb-edu, python 
    mixed_dataset = interleave_datasets(hf_datasources, probs, seed=42, stopping_strategy="all_exhausted")
    
    return mixed_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=42)


def get_sft_dataset():
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split='train', streaming=True)#, token=hf_token)
    return ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=42)
    

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