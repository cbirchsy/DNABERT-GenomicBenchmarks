import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from transformers import AutoTokenizer
from genomic_benchmarks.dataset_getters.pytorch_datasets import GenomicClfDataset
import pandas as pd
import numpy as np

def kmerize(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

class GenomicBenchmarksDataset(Dataset):
    def __init__(self, dataset, tokenizer, k, tokenizer_kwargs, padding=False, tokenize_for_dnabert=True):
        super().__init__()
        self.seqs = dataset['seq']
        self.labels = torch.Tensor(dataset['label']).long()
        self.tokenizer = tokenizer
        self.k = k
        self.tokenizer_kwargs = tokenizer_kwargs
        self.padding = padding
        self.tokenize_for_dnabert = tokenize_for_dnabert
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.tokenize_for_dnabert:
            seq = kmerize(self.seqs[idx], self.k)
        else:
            seq = self.seqs[idx]
        tokenized = self.tokenizer(seq, **self.tokenizer_kwargs)
        x = torch.Tensor(tokenized['input_ids']).long()
        if self.padding:
            attention_mask = torch.Tensor(tokenized['attention_mask']).long()
            return x, attention_mask, self.labels[idx]
        else:
            return x, self.labels[idx]

class PreTokenizedGenomicBenchmarksDataset(Dataset):
    def __init__(self, tokenized_dataset, padding=False):
        super().__init__()
        self.input_ids = torch.Tensor(tokenized_dataset['input_ids']).long()
        self.labels = torch.Tensor(tokenized_dataset['label']).long()
        self.padding = padding
        if self.padding:
            self.attention_mask = torch.Tensor(tokenized_dataset['attention_mask']).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.padding:
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        else:
            return self.input_ids[idx], self.labels[idx]

class GenomicBenchmarksDataModule(pl.LightningDataModule):
    def __init__(self, k, tokenizer_path, dataset_name, batch_size=32, num_workers=0, pin_memory=True, pretokenize=True, padding=False, tokenize_for_dnabert=True):
        super().__init__()
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pretokenize = pretokenize
        self.padding = padding
        self.tokenize_for_dnabert = tokenize_for_dnabert
        if self.tokenize_for_dnabert:
            self.model_max_length = 512
        else:
            self.model_max_length = 1002
        
        self._configure_dataset(dataset_name)
        
        self.tokenizer_kwargs={
            'truncation':True,
            'max_length':self.max_length,
        }
        if self.padding:
            self.tokenizer_kwargs['padding']='max_length'

        if self.pretokenize:
            print("Pre-tokenizing sequences...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            self.dataset = self.dataset.map(self._tokenize, batched=True)
            print("Done")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            
    def _tokenize(self, dataset):
        if self.tokenize_for_dnabert:
            return self.tokenizer([kmerize(seq, self.k) for seq in dataset['seq']], **self.tokenizer_kwargs)
        else:
            return self.tokenizer(dataset['seq'], **self.tokenizer_kwargs)
    
    def _configure_dataset(self, dataset_name):
        df_train = pd.DataFrame(
            data=list(
                GenomicClfDataset(dataset_name, split='train', version=0)
            ), 
            columns=['seq', 'label']
        )
        ds_train = datasets.Dataset.from_pandas(df_train)
        
        df_test = pd.DataFrame(
            data=list(
                GenomicClfDataset(dataset_name, split='test', version=0)
            ), 
            columns=['seq', 'label']
        )
        ds_test = datasets.Dataset.from_pandas(df_test)
        
        self.max_length = min(self.model_max_length, max(max([len(seq) for seq in df_train['seq']]), max([len(seq) for seq in df_test['seq']])))

        self.dataset = datasets.DatasetDict()
        self.dataset['train'] = ds_train
        self.dataset['test'] = ds_test
            
    def setup(self, stage=None):
        if self.pretokenize:
            self.dataset_train = PreTokenizedGenomicBenchmarksDataset(self.dataset['train'], padding = self.padding)
            self.dataset_test = PreTokenizedGenomicBenchmarksDataset(self.dataset['test'], padding = self.padding)
        else:
            self.dataset_train = GenomicBenchmarksDataset(self.dataset['train'], self.tokenizer, self.k, self.tokenizer_kwargs, padding = self.padding, tokenize_for_dnabert=self.tokenize_for_dnabert)
            self.dataset_test = GenomicBenchmarksDataset(self.dataset['test'], self.tokenizer, self.k, self.tokenizer_kwargs, padding = self.padding, tokenize_for_dnabert=self.tokenize_for_dnabert)

    def train_dataloader(self):
        return self._data_loader(self.dataset_train)

    def val_dataloader(self):
        return self._data_loader(self.dataset_test, shuffle=False) # there is no val set

    def test_dataloader(self):
        return self._data_loader(self.dataset_test, shuffle=False)

    def _data_loader(self, dataset, shuffle = True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )