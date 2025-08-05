# CPSVD: Enhancing Large Language Model Compression via Column-Preserving Singular Value Decomposition

This repository provides the official implementation of **CPSVD**, a novel method for compressing large language models (LLMs) by combining column selection and SVD-based low-rank approximation. CPSVD is designed to preserve important structural information in the weight matrices, enabling efficient compression without sacrificing performance.

## 🚀 Quick Start

### 1. Installation

Create and activate a conda environment:

```bash
conda create -n cpsvd python=3.9
conda activate cpsvd
```

Clone the repository:

```bash
git clone https://github.com/Pupu792/CPSVD.git
cd CPSVD
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Example
You can run the main script with the following command:
```bash
python SVDLLM.py --model /models/Llama-7b --ratio 0.4 --step -1 --trunc_rank_method cos --t 0.1 --matrices_optimized --eval_ppl --eval_zero_shot --cuda_devices 0
```

By default, the results are saved in a format compatible with the ``` transformers``` library.

**Note:** If you need to save the decomposed version, set the `--step` argument to `0` and modify the `transformers` library to version `4.35.2`. For detailed instructions, please refer to the original project: [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM).


## 📊 Evaluation

For evaluating zero-shot tasks, we adopt the methods used in the [wanda](https://github.com/locuslab/wanda) repository.

---
## 🙏 Acknowledgments

This repository is built upon the foundational work of [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM).