# ATACompass

**ATACompass** is a computational framework for **cross-species identification of cell identities from single-cell ATAC-seq (scATAC-seq) data without relying on gene annotations**.

By leveraging **sequence-based regulatory conservation and large-scale genomic sequence models**, ATACompass enables the comparison of chromatin accessibility landscapes across species even when genome annotations are incomplete or inconsistent.

The framework integrates:

- scATAC-seq data preprocessing
- cross-species regulatory sequence conservation analysis
- sequence-based large language models for genomic sequences

to infer and compare cell identities across species.

<img width="1009" height="644" alt="5e6f545c-63ee-4b36-a260-f6cc79049f49" src="https://github.com/user-attachments/assets/8b236ba2-d8c1-4252-81a9-5adc49a898a0" />


------

# Overview

Single-cell ATAC-seq provides insights into chromatin accessibility and regulatory landscapes at single-cell resolution. However, **cross-species comparison of scATAC-seq data remains challenging** due to:

- differences in genome annotation quality
- species-specific gene models
- limited conservation of gene structures

Traditional approaches rely on **gene activity scores or orthologous gene mapping**, which may introduce bias in evolutionary comparisons.

**ATACompass addresses this limitation by focusing on sequence-level regulatory conservation and sequence-based representations of chromatin accessibility signals.**

------

# Installation

Clone the repository:

```
git clone https://github.com/LalicJ/ATACompass.git
cd ATACompass
```

Create a conda environment:

```
conda create -n atacompass python=3.9
conda activate atacompass
```

Install dependencies:

```
pip install -r requirements.txt
```

For the training and evaluation pipeline:

```
conda env create -f train_eval_code/environment.yml
conda activate atacompass
```

------

# Data Preprocessing

Preprocessing scripts for preparing genomic sequences are located in:

```
preprocessing/
```

Example:

```
gen_csv_orders_of_base_seqs.py
clean_csv_orders_of_base_seqs.py
```

These scripts convert genomic sequences into tokenized representations suitable for downstream sequence modeling.

------

# Cross-Species Sequence Conservation Analysis

Scripts for cross-species regulatory sequence conservation analysis are located in:

```
cross-species sequence conservation/
```

Key steps include:

1. Peak calling from ATAC-seq data
2. Sequence extraction
3. BLAT-based sequence alignment
4. Filtering conserved regulatory regions

Main pipeline:

```
run_pipeline.sh
```

# Sequence Model Training

Model training and evaluation code is located in:

```
train_eval_code/
```

This module includes:

- dataset preparation
- model configuration
- training pipeline
- evaluation scripts

Key training scripts:

```
train_base_seqs_llm_stage1.py
train_base_seqs_llm_stage2.py
```

Example training command:

```
bash run_train_base_seqs_llm.sh
```

------

# Model Architecture

ATACompass integrates **sequence-based genomic language models**, including architectures inspired by:

- transformer models
- Hyena sequence models
- attention-based architectures for long genomic sequences

Model implementations are located in:

```
train_eval_code/models/
```

------

# Configuration System

Experiment configuration files are located in:

```
train_eval_code/configs/
```

These YAML configuration files define:

- dataset settings
- model architecture
- optimizer and scheduler
- training pipelines

------

# Evaluation

Evaluation scripts include:

```
test_base_seqs_llm.py
test_base_seqs_llm_with_cluster.py
```

These scripts evaluate the trained sequence models for cross-species regulatory representation learning.

