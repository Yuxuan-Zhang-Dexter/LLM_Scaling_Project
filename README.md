
# Language Model Training Pipeline

This repository contains a training pipeline for fine-tuning language models using the Transformers library. It includes scripts for dataset preparation, model training, and monitoring.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Prepare and Cache Dataset](#1-prepare-and-cache-dataset)
  - [2. Train the Model](#2-train-the-model)
- [Configuration](#configuration)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Dependencies](#dependencies)

---

## Project Overview

This project enables efficient fine-tuning of large language models using **Hugging Face Transformers** and **DeepSpeed**. It also supports:
- Dynamic dataset preparation and caching.
- Training with advanced optimization techniques like DeepSpeed Zero Init.
- Real-time monitoring via WandB.

---

## Features

- **Dataset Preparation**: Tokenize and preprocess large datasets for model training.
- **DeepSpeed Integration**: Efficient training of large models with Zero optimization.
- **WandB Integration**: Tracks training metrics and visualizations.
- **Configuration-Driven**: Define training parameters and paths via JSON files or CLI arguments.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional Linux tools (optional, e.g., for GPU monitoring):
   ```bash
   bash tool_install.sh
   ```

---

## Usage

### 1. Prepare and Cache Dataset

Use `load_data.py` to preprocess and tokenize your dataset. The script supports custom tokenizers and sequence packing.

#### Example:
```bash
python load_data.py \
  --tokenizer_path /workspace/ML_team/llama_tokenizer_1b \
  --dataset_name DKYoon/SlimPajama-6B \
  --output_dir ./datasets_pack_full/tokenized_data \
  --num_proc 16
```

#### Key Arguments:
- `--tokenizer_path`: Path to your custom tokenizer directory.
- `--dataset_name`: Dataset name from Hugging Face hub (e.g., `DKYoon/SlimPajama-6B`).
- `--output_dir`: Directory to save the tokenized dataset.
- `--max_seq_len`: Maximum sequence length for tokenization (default: 1024).
- `--num_proc`: Number of processes for parallel tokenization (default: 8).

---

### 2. Train the Model

Once the dataset is prepared, use `train.py` to fine-tune your model. Training parameters and paths are specified in a configuration file.

#### Running Training with DeepSpeed:

DeepSpeed is used to optimize large-scale model training. To run the training with DeepSpeed, use the following command:

```bash
deepspeed train.py --config config.json
```

#### Explanation of the Command:

- **`deepspeed`**: Invokes the DeepSpeed runtime to manage distributed training and memory optimizations.
- **`train.py`**: The training script, which initializes the model, tokenizer, and datasets based on the `config.json` file.
- **`--config config.json`**: Specifies the path to the configuration file containing all training parameters.

#### What to Expect:
- **Initialization Logs**:
  - DeepSpeed initialization with `Zero Init` (if enabled).
  - GPU allocation and compute type (`fp16` or `bf16`).
- **Real-Time Metrics**:
  - Progress logs and metrics will be reported to WandB, if configured.
- **Model Checkpoints**:
  - Checkpoints will be saved periodically as defined in `save_steps` and `checkpoint_output_dir`.

---

## Configuration

### `load_data.py` Configuration:
Customize via CLI arguments:
- `--tokenizer_path`: Path to tokenizer.
- `--dataset_name`: Hugging Face dataset name.
- `--max_seq_len`: Maximum token length (default: 1024).
- `--num_proc`: Number of processes for tokenization.

```
python load_data.py \\
  --tokenizer_path /workspace/ML_team/llama_tokenizer_1b \\
  --dataset_name DKYoon/SlimPajama-6B \\
  --output_dir ./datasets_pack_full/tokenized_data \\
  --num_proc 16
```

### `train.py` Configuration:
Edit `config.json` to:
- Set model paths, optimizer settings, and training hyperparameters.
- Configure DeepSpeed optimizations (`use_zero3`, gradient accumulation).
- Enable WandB for real-time monitoring.

Example `config.json`:
```json
{
    "seed": 42,
    "cuda_visible_devices": "0,1",
    "master_addr": "localhost",
    "master_port": "9994",
    "data_path": "./datasets_pack_full/tokenized_data",
    "model_path": "./configs/model_configs/llama_190M_config.json",
    "checkpoint_output_dir": "./model_checkpoints",
    "deepspeed_config": "./configs/deepspeed_configs/test_ds_zero2_config.json",
    "tokenizer_config": "./configs/llama_tokenizer_configs",
    "logging_dir": "./logs",
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "save_steps": 50000,
    "logging_steps": 20,
    "eval_steps": 10000,
    "num_epoch": 1,
    "batch_size": 16,
    "gradient_checkpointing": false,
    "fp16": true,
    "bf16": false,
    "learning_rate": 0.0003,
    "gradient_accumulation": 1,
    "weight_decay": 0.00003,
    "save_total_limit": 2,
    "warmup_steps": 500,
    "use_zero3": true,
    "lr_scheduler_type": "cosine",
    "vis_app": "wandb",
    "final_model": {
        "path": "./final_model/target_model_config",
        "tokenizer_path": "./final_model/target_tokenizer_config"
    },
    "wandb": {
        "key": "<your-wandb-key>",
        "project_name": "llama-training",
        "entity_name": "<your-wandb-entity>"
    }
}
```

```
deepspeed train.py --config config.json
```

---

## Monitoring and Debugging

### GPU Monitoring
Install and use tools like `nvtop`:
```bash
sudo apt install nvtop
```

### WandB Metrics
Track training progress:
- Log in via API key in the configuration.
- View real-time metrics on the WandB dashboard.

---

## Dependencies

### Python Dependencies
Install via `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Linux Tools
Run `tool_install.sh` to install:
- **`nvtop`**: Monitor GPU usage.
- **Other tools**: Extend as needed.

---

## Contribution

Feel free to fork the repository and submit PRs for improvements. Issues and feature requests are welcome!
