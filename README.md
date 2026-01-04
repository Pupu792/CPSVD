# Duo-SVD

This repository provides the official implementation of the article: **Beyond Uniform SVD: Dual-Level Optimization across Columns and Modules for LLM Compression**.

## 🚀 Quick Start

### 1. Installation

Create and activate a conda environment, then install the required dependencies.

```bash
# Create environment
conda create -n duo-svd python=3.9.20
conda activate duo-svd

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage Pipeline

The compression workflow consists of three preparation steps followed by evaluation.

#### Step 1: Get Hessian Matrix
Collect the Hessian matrix corresponding to the weight matrix.

```bash
python Duo-SVD.py \
    --step 1 \
    --model /path/to/source_model \
    --dataset wikitext2 \
    --save_path /path/to/save_hessian
```

#### Step 2: Calculate Module Sensitivity
Calculate the perturbation for each module under the target compression ratio.

```bash
python Duo-SVD.py \
    --step 2 \
    --model /path/to/source_model \
    --hessian_mat_path /path/to/save_hessian \
    --dataset wikitext2 \
    --whitening_nsamples 32 \
    --model_seq_len 2048 \
    --ratio 0.8 \
```
> **Note:** Replace `0.8` with your target compression ratio. Ensure you verify where the sensitivity results are saved (referenced as `sensitivity_path` in the next step).

#### Step 3: Compress Model
Perform the model compression.

```bash
python Duo-SVD.py \
    --step 3 \
    --model /path/to/source_model \
    --hessian_mat_path /path/to/save_hessian \
    --ratio 0.8 \
    --strategy dp \
    --sensitivity_path /path/to/save_sensitivity \
    --save_path /path/to/save_compressed_model
```

#### Step 4-6: Evaluation
Evaluate the compressed model generated in Step 3.

**Evaluate Perplexity (Step 4)**
```bash
python Duo-SVD.py \
    --step 4 \
    --model_path /path/to/save_compressed_model
```

**Evaluate Inference Performance (Step 5)**
Measure latency and throughput with specific generation settings.
```bash
python Duo-SVD.py \
    --step 5 \
    --model_path /path/to/save_compressed_model \
    --prefilling_len 256 \
    --gen_seq_len 64 \
    --eval_batch_size 24
```

**Evaluate Downstream Tasks (Step 6)**
```bash
python Duo-SVD.py \
    --step 6 \
    --model_path /path/to/save_compressed_model
```

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE).