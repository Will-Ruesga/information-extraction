# CADEC Named Entity Recognition (NER) with BERT, CRF, and BiLSTM-CRF

This repository contains code to train and evaluate Named Entity Recognition (NER) models on the [CSIRO Adverse Drug Event Corpus (CADEC)](https://data.csiro.au/dap/landingpage?pid=csiro:9787).

Three architectures are implemented:
1. **BERT Baseline** – Fine-tuned `bert-base-cased` for token classification.
2. **BERT + CRF** – Adds a Conditional Random Field layer on top of the baseline BERT.
3. **BERT + BiLSTM + CRF** – Adds a bidirectional LSTM layer before the CRF head.

All models perform sequence labeling for the entity types:
```
O, B-ADR, I-ADR, B-Disease, I-Disease, B-Drug, I-Drug, B-Finding, I-Finding, B-Symptom, I-Symptom
```
---

## Installation
```bash
git clone <repo_url>
cd <repo_name>
pip install -r requirements.txt
```
