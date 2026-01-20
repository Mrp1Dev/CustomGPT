import torch
from torch.utils.data import IterableDataset
import random
import pyarrow.parquet as pq
import sentencepiece as spm

class ParquetDataset(IterableDataset):
    def __init__(self, files, B, T, tokenizer: spm.SentencePieceProcessor,
                 append_eos = False, loop_around = True, shuffle_rgs = False, seed = 42):
        self.files = files
        self.B = B
        self.T = T
        self.tokenizer = tokenizer
        self.block_size = B*T + 1
        self.append_eos = append_eos
        self.shuffle_rgs = shuffle_rgs
        self.loop_around = loop_around
        self.seed = seed 
        self.epoch = 0
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        buffer = []
        
        while True:
            for file_idx, file_path in enumerate(self.files):
                pf = pq.ParquetFile(file_path)
                num_rg_groups = pf.num_row_groups
                my_rg_ids = list(range(worker_id, num_rg_groups, num_workers))
                if self.shuffle_rgs:
                    random.seed(self.seed + self.epoch + worker_id + file_idx)
                    random.shuffle(my_rg_ids)
                for rg_id in my_rg_ids:
                    row_group = pf.read_row_group(rg_id)
                    text_rows = row_group.column('text').to_pylist()
                    for text in text_rows:
                        tokens = self.tokenizer.Encode(text)
                        if self.append_eos:
                            tokens = tokens + [self.tokenizer.eos_id()]
                        buffer.extend(tokens)
                        
                        while len(buffer) >= self.block_size:
                            token_chunk = buffer[:self.block_size] # first self.block_size tokens
                            data = torch.tensor(token_chunk, dtype=torch.long)
                            xs = data[:-1].view(self.B, self.T)
                            ys = data[1:].view(self.B, self.T)
                            yield xs, ys
                            buffer = buffer[self.block_size:]
            if self.loop_around == False:
                break
            self.epoch = self.epoch + 1                    

        