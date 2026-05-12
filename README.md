# Int-ResGAT

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.7.0-orange.svg)](https://pyg.org/)

Official implementation of **Int-ResGAT**, the Integrated Residue-Level Graph Attention Network for multi-label enzyme function prediction.

This repository accompanies the manuscript:

> **Integrated Residue-Level Graph Attention Network for Multi-Label Enzyme Function Prediction**

Int-ResGAT integrates completed experimental enzyme crystal structures, ESM-3 residue embeddings, and a hierarchical residue-level graph attention network to predict Enzyme Commission (EC) functions. The framework is designed for enzyme annotation under incomplete structural coverage, long-tailed EC distributions, and rare or divergent enzyme classes.

## Overview

Int-ResGAT has two main components:

1. **Enzyme crystal structure completion**
   - Downloads experimental PDB/mmCIF structures and AlphaFold or SWISS-MODEL predicted structures.
   - Aligns predicted full-length structures to experimental fragments.
   - Preserves experimental backbone coordinates where available and fills unresolved regions.
   - Optionally applies side-chain repacking and local geometry relaxation.

2. **HE-ResGATConv model**
   - Represents each residue as a graph node initialized with 1536-dimensional ESM-3 embeddings.
   - Builds spatial residue graphs using C-alpha distances.
   - Uses GATv2Conv layers with residual connections and distance-aware edge attributes.
   - Aggregates multi-scale graph representations with mean and max pooling across graph layers.
   - Uses sigmoid multi-label prediction with focal BCE loss and positive-class weighting to improve performance on imbalanced EC categories.

In the manuscript benchmark, Int-ResGAT achieved:

| Metric | Value |
| --- | ---: |
| Top-1 Accuracy | 0.9540 |
| Macro-F1 | 0.7251 |
| Macro-Precision | 0.7221 |
| Macro-Recall | 0.7523 |

The reported results were obtained on a preprocessed dataset of 142,824 three-dimensional enzyme representations covering 1,765 EC categories.

## Repository Structure

```text
Int-ResGAT/
|-- data/
|   |-- database.csv              # Metadata table used by the default configuration
|   `-- *.py / *.txt / *.xml      # Optional local data preparation files
|-- dataloader/
|   |-- dataloader.py             # Data initialization pipeline
|   `-- utils.py                  # Structure download, completion, embedding and graph utilities
|-- train/
|   |-- loss.py                   # Focal Loss implementation
|   |-- models.py                 # HE_ResGATConv model
|   |-- train.py                  # Training and evaluation loop
|   `-- utils.py                  # Dataset and dataloader utilities
|-- config.yaml                   # Runtime configuration
|-- dist_train.sh                 # Multi-GPU torchrun launcher
|-- main.py                       # Main entry point
|-- requirements.txt              # Python dependencies
`-- LICENSE
```

## Installation

The code was developed with Python 3.10.12, PyTorch 2.3.1 and PyTorch Geometric 2.7.0.

```bash
git clone https://github.com/LycrsLOL/Int-ResGAT.git
cd Int-ResGAT

conda create -n int-resgat python=3.10 -y
conda activate int-resgat

pip install -r requirements.txt
```

For GPU execution, install the PyTorch and PyTorch Geometric wheels that match your CUDA driver before installing the remaining dependencies if needed.

### Optional External Tools

The structure-completion pipeline can use the following tools when available:

- **SCWRL4** for side-chain repacking.
- **OpenMM** for local geometry relaxation.

If these tools are not installed, the pipeline falls back to the available intermediate structure, but exact reproduction of the manuscript preprocessing should use the same structure-completion setup.

## Data

The default metadata table is provided at:

```text
data/database.csv
```

Required columns:

| Column | Description |
| --- | --- |
| `pdb_id` | PDB identifier for the experimental structure, when available |
| `uniprot_id` | UniProt accession |
| `ec_numbers` | EC annotation; multiple labels may be separated by semicolons |
| `active_sites` | Optional active-site residue positions used for analysis |
| `homology_cluster` | Optional homologous-family or sequence-cluster identifier used for leakage-controlled train/test splitting |

For the manuscript benchmark, training and test proteins should be separated by homologous groups. You can provide the group assignments either as a `homology_cluster` column in `data/database.csv` or as a separate CSV/TSV file configured by `train.homology_cluster_file`. The external file must contain `uniprot_id` plus one cluster column such as `homology_cluster`, `cluster_id`, `mmseqs_cluster` or `cdhit_cluster`.

The complete dataset supporting the manuscript is archived on Zenodo:

**https://doi.org/10.5281/zenodo.20123394**

The dataset was constructed from Swiss-Prot EC annotations, SIFTS residue-level sequence-structure mappings, experimental PDB structures, and AlphaFold Protein Structure Database entries.

## Configuration

Edit `config.yaml` before running the pipeline. In particular, update paths that are specific to the original training environment:

```yaml
database_path: ./data/database.csv
save_dir: ./output

initialize:
  model_path: ./models/esm3_sm_open_v1.pth
  max_edge_distance: 10.0
```

Important options:

| Option | Description |
| --- | --- |
| `database_path` | Metadata CSV file |
| `save_dir` | Directory for downloaded structures, embeddings, graphs and checkpoints |
| `initialize.model_path` | Local ESM-3 checkpoint path |
| `initialize.max_edge_distance` | Residue graph distance cutoff in Angstrom |
| `train.batch_size` | Batch size per process/GPU |
| `train.learning_rate` | Initial Adam learning rate |
| `train.hidden_dim` | Hidden dimension of HE-ResGATConv |
| `train.num_layers` | Number of graph attention layers |
| `train.heads` | Number of attention heads |
| `train.focal_loss_gamma` | Focusing parameter for Focal Loss |
| `train.split_strategy` | `homology` for manuscript experiments; `random` only for debugging |
| `train.homology_group_column` | Cluster column used by the homology split |
| `train.homology_cluster_file` | Optional cluster assignment table keyed by `uniprot_id` |
| `train.prediction_threshold` | Sigmoid threshold for multi-label metrics |

If you use the installed `esm` Python package directly, make sure any local ESM source-path setting in `dataloader/dataloader.py` is consistent with your environment.

## Usage

### 1. Initialize Structures, Embeddings and Graphs

```bash
python main.py --mode initialize --config config.yaml
```

This step performs:

1. Structure download.
2. Experimental/predicted structure completion.
3. ESM-3 residue embedding extraction.
4. Embedding validation.
5. PyTorch Geometric graph construction.

Generated files are written under `save_dir`, including:

```text
output/
|-- crystal/
|-- predicted/
|-- complete/
|-- embedding/
`-- graph/
```

### 2. Train on One GPU

```bash
python main.py --mode train --config config.yaml
```

The training code now treats EC prediction as a true multi-label problem: EC strings are split on semicolons, labels are encoded as multi-hot vectors, the classifier is optimized with multi-label focal BCE loss, and metrics are computed from sigmoid probabilities. Reported training logs include Top-1/Top-5 hit rate, exact subset accuracy, Macro/Micro-F1, Fmax, AUROC and AUPRC when computable.

### 3. Train with Distributed Data Parallel

Edit `CUDA_VISIBLE_DEVICES` in `dist_train.sh`, then run:

```bash
bash dist_train.sh
```

The launcher uses `torchrun` and automatically sets the number of processes from the selected GPU list.

### 4. Resume Training

```bash
python main.py --mode train --config config.yaml --resume /path/to/checkpoint_latest.pth
```

## Outputs

During training, Int-ResGAT writes:

- log files to `logs/`;
- the latest checkpoint to `<save_dir>/checkpoint_latest.pth`;
- the best model by Macro-F1 to `<save_dir>/best_model/`.

Large checkpoint files, graph tensors and model weights can exceed GitHub's 100 MB file-size limit. If these artifacts are published in this repository, track them with Git LFS before committing:

```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
```

For archival release, Zenodo or another data repository is recommended for large pretrained checkpoints and generated graph datasets.

## Reproducing the Manuscript Experiments

To reproduce the paper workflow:

1. Download or prepare the dataset described in the Zenodo archive.
2. Add homologous-family cluster assignments to `data/database.csv` or configure `train.homology_cluster_file`.
3. Set `database_path`, `save_dir`, and `initialize.model_path` in `config.yaml`.
4. Run the initialization pipeline to generate completed structures, ESM-3 embeddings and residue graphs.
5. Train Int-ResGAT with the hyperparameters in `config.yaml`.
6. Evaluate the resulting logs against the metrics reported in the manuscript.

Default hyperparameters from the manuscript configuration:

| Hyperparameter | Value |
| --- | ---: |
| Input dimension | 1536 |
| Hidden dimension | 1024 |
| Graph layers | 4 |
| Attention heads | 8 |
| Dropout | 0.5 |
| Weight decay | 0.0005 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Maximum epochs | 300 |
| Minimum epochs | 100 |
| LR patience | 10 |
| Early-stopping patience | 20 |
| Random seed | 42 |

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contact

For questions about the code or manuscript, please open an issue in this repository.
