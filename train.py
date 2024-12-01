import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import wandb
import os
import deepspeed
import json
from utilities import seed_everything, check_cuda_availability, determine_compute_dtype_and_attention, count_parameters

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config(config_file):
    """
    Load training configuration from a JSON file.
    """
    with open(config_file, "r") as f:
        return json.load(f)

def initialize_model(config_path, attn_implementation, use_zero3=False):
    """
    Initialize the model with the provided configuration path and attention implementation settings.
    Conditionally use DeepSpeed's zero.Init based on `use_zero3`.
    """
    if use_zero3:
        print("Initializing model with DeepSpeed zero.Init...")
        with deepspeed.zero.Init():
            config = AutoConfig.from_pretrained(config_path)
            model = AutoModelForCausalLM.from_config(
                config,
                attn_implementation=attn_implementation["attn_implementation"],
                torch_dtype=attn_implementation["compute_dtype"]
            )
    else:
        print("Initializing model directly without DeepSpeed zero.Init...")
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attn_implementation["attn_implementation"],
            torch_dtype=attn_implementation["compute_dtype"]
        )

    print(f"Total number of trainable parameters: {count_parameters(model):,}")
    return model

def main(config_file):
    # Load configuration
    config = load_config(config_file)

    # Environment setup
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    os.environ["MASTER_ADDR"] = config["master_addr"]
    os.environ["MASTER_PORT"] = config["master_port"]

    # Initialize random seed and check CUDA availability
    seed_everything(config["seed"])
    check_cuda_availability()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Wandb
    wandb.login(key=config["wandb"]["key"])
    wandb.init(project=config["wandb"]["project_name"], entity=config["wandb"]["entity_name"])

    # Determine compute dtype and attention implementation
    attn_implementation = determine_compute_dtype_and_attention()
    print(f"Compute type: {attn_implementation['compute_dtype']}")

    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_config"])
    tokenizer.pad_token = tokenizer.eos_token
    model = initialize_model(config["model_path"], attn_implementation, use_zero3=config["use_zero3"])
    model.to(device)
    model.config.use_cache = False

    # Load datasets
    dataset_train = load_from_disk(os.path.join(config["data_path"], "train"))
    dataset_eval = load_from_disk(os.path.join(config["data_path"], "validation"))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["checkpoint_output_dir"],
        evaluation_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_epoch"],
        weight_decay=config["weight_decay"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        report_to=config["vis_app"],
        logging_dir=config["logging_dir"],
        logging_steps=config["logging_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        save_steps=config["save_steps"],
        deepspeed=config["deepspeed_config"] if config["use_zero3"] else None,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        warmup_steps=config["warmup_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        save_strategy=config["save_strategy"],
        save_total_limit=config["save_total_limit"],
    )

    # Custom collator for data handling
    def custom_collator(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=custom_collator,
    )

    # Start training
    print("Start training...")
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained(config["final_model"]["path"])
    tokenizer.save_pretrained(config["final_model"]["tokenizer_path"])
    print("Saved final model and tokenizer.")

    wandb.finish()


if __name__ == "__main__":
    import argparse

    # Parse the configuration file from command-line arguments
    parser = argparse.ArgumentParser(description="Train a language model using a configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    main(args.config)
