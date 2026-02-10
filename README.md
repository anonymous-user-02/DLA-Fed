# DLA-Fed: Dynamic Layer-wise Adaptation for Label-Heterogeneous Federated Learning

This repository provides the official implementation of DLA-Fed, a federated learning framework designed to address label heterogeneity across distributed clients in medical imaging, with a primary focus on chest X-ray diagnosis. DLA-Fed enables collaborative learning without label sharing by dynamically regularizing and aggregating shared feature extractor layers, while allowing client-specific classifier heads to be locally fine-tuned.

The framework supports experiments on publicly available chest X-ray datasets and enables fair comparison with several personalized and label-heterogeneous federated learning baselines, including our prior work.

---

## Dataset Preparation

The `dataset/` directory contains scripts to generate datasets for federated learning experiments. Each dataset has a dedicated script that prepares client-wise data splits suitable for label-heterogeneous settings.

### Available Dataset Scripts

- generate_CheXpert.py
- generate_MIMIC.py
- generate_NIHChestXray.py

### Generating a Dataset

From the `dataset/` directory, run:

    python3 <SCRIPT_NAME>

Example:

    python3 generate_NIHChestXray.py

Within each dataset script, the following parameters can be modified:

- Number of clients
- Number of runs
- Dataset split configuration
- Output directory paths

All parameters are defined directly inside the corresponding script.

---

## Running Experiments

All federated learning experiments are executed from the `system/` directory.

### Main Components

- main.py: Core federated learning framework
- run.sh: Script to launch experiments sequentially

### Running Experiments

    cd system
    ./run.sh

Hyperparameters for each method and dataset can be adjusted directly inside `run.sh`.

---

## Example Command

Below is an example command for running a single experiment using **DLA-Fed**. Similar commands are used for other methods and datasets.

    nohup python3 main.py \
      -algo DLAFed \
      -lr 0.01 \
      -tmp 7.5 \
      -lam 0.95 \
      -m effnet \
      -mn effnet \
      -nc 3 \
      -data nihchestxray \
      -t 5 \
      -go experiment \
      -gpu 0 \
      > DLAFed_nihchestxray_3_effnet.log 2>&1 &

---

## Supported Methods

- DLA-Fed (proposed)
- FD-Fed
- FedBABU
- FedPer
- FedRep
- FedPav
- Local training

---

## Notes

- Experiments are executed asynchronously using `nohup`.
- Training logs are saved to `.log` files for later inspection.
- GPU selection is controlled via the `-gpu` argument.
