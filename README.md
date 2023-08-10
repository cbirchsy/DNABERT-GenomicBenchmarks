## This repo is for running experiments on Genomic Benchmarks datasets with DNABERT or NucleotideTransformer.

## To get started

Install pytorch:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install transformers from source (needed for NucleotideTransformer) and additional requirements with pip:
```
pip install --upgrade git+https://github.com/huggingface/transformers.git
pip install lightning datasets peft wandb genomic_benchmarks hydra-core hydra-colorlog rootutils
```

To run an experiment do:
```
python src/train.py experiment=<model>/<dataset>
```

where `<model> = DNABERT or NucleotideTransformer` and `<dataset>` is any of:

- `dummy_mouse_enhancers_ensembl`
- `human_enhancers_cohn`
- `human_nontata_promoters`
- `demo_coding_vs_intergenomic_seqs`
- `demo_human_or_worm`
- `human_enhancers_ensembl`
- `human_ocr_ensembl`
- `human_ensembl_regulatory`

Change parameters by editing the experiment configs or by overriding on command line e.g. 
```
python src/train.py experiment=NucleotideTransformer/human_enhancers_ensembl data.batch_size=16 data.padding=True model.use_attention_mask=True
```


## Details on configurable arguments

GenomicsBenchmarksDataModule:
- `k`: The value of k used when generating k-mers for DNABERT tokenizer (ignored if `tokenize_for_dnabert=False`).
- `tokenizer_path`: Path to HuggingFace tokenizer e.g "InstaDeepAI/nucleotide-transformer-2.5b-multi-species".
- `dataset_name`: Name of the Genomic Benchmarks dataset to use e.g. "human_nontata_promoters".
- `pretokenize`: If `True` the sequences will be tokenized before training rather than on the fly.
- `padding`: Whether to pad the sequences and return attention mask in the dataloader. In the model `use_attention_mask` should be set as same.
  - For DNABERT this should be set to `True` if the dataset has variable length sequence and `batch_size>1`.
  - For NucleotideTransformer this should be set to `True` if `batch_size>1` since the tokenizer doesn't yield fixed length of tokenized sequences depending on the content.
- `tokenize_for_dnabert`: Whether to tokenize for DNABERT (required since the seqs need k-merizing before tokenizing whereas NT tokenizer works on raw seqs).

DNABERTClassifier/NucleotideTransformerClassifier:
- `optimizer`: A torch.optim config.
- `scheduler`: A torch.optim.lr_scheduler config.
- `model_path`: The path to HuggingFace model e.g "InstaDeepAI/nucleotide-transformer-2.5b-multi-species".
- `num_labels`: Sets the number of output nodes in classifier head. Must be equal to the number of classes in the dataset.
- `use_attention_mask`: Whether to make use of the attention mask in the forward pass. Must be used if `padding = True` in datamodule.
- `peft`: (NucleotideTransformerClassifier only) whether to use IA3 parameter efficient fine-tuning method decribed in the NucleotideTransformer paper.

