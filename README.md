## This repo is for running experiments on Genomic Benchmarks datasets with DNABERT or NucleotideTransformer.

# To get started

I install my env like this:

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



