import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import argparse
from utilities import seed_everything, print_dataset_shape

# Seed for reproducibility
seed_everything(42)

def prepare_and_cache_dataset(args):
    """
    Tokenizes and caches datasets based on provided command-line arguments.
    Args:
        args: Command-line arguments parsed via argparse.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, cache_dir=args.cache_dir)
    
    # Load datasets
    train_subset = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir, download_mode='reuse_cache_if_exists')
    validation_subset = load_dataset(args.dataset_name, split="validation", cache_dir=args.cache_dir, download_mode='reuse_cache_if_exists')
    test_subset = load_dataset(args.dataset_name, split="test", cache_dir=args.cache_dir, download_mode='reuse_cache_if_exists')

    # Tokenization function with sequence packing and truncation
    def pack_sequences_truncate(batch, max_length=args.max_seq_len):
        input_ids, attention_masks = [], []
        current_chunk = {"input_ids": []}
        current_length = 0

        for tokens in batch['text']:
            tokenized = tokenizer(tokens, truncation=True, max_length=max_length, padding=False, return_overflowing_tokens=True, stride=int(max_length * 0.2))
            for sequence in tokenized['input_ids']:
                if (current_length + len(sequence) + 1) <= max_length:
                    current_chunk["input_ids"].extend(sequence + [tokenizer.eos_token_id])
                    current_length += len(sequence) + 1
                else:
                    input_ids.append(current_chunk["input_ids"][:max_length])
                    attention_masks.append([1] * max_length)
                    current_chunk = {"input_ids": sequence[:max_length - 1] + [tokenizer.eos_token_id]}
                    current_length = len(current_chunk["input_ids"])
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_masks), "labels": torch.tensor(input_ids)}

    # Apply tokenization
    train_tokenized_dataset = train_subset.map(pack_sequences_truncate, batched=True, remove_columns=['text', 'meta', '__index_level_0__'], num_proc=args.num_proc)
    validation_tokenized_dataset = validation_subset.map(pack_sequences_truncate, batched=True, remove_columns=['text', 'meta', '__index_level_0__'], num_proc=args.num_proc)
    test_tokenized_dataset = test_subset.map(pack_sequences_truncate, batched=True, remove_columns=['text', 'meta', '__index_level_0__'], num_proc=args.num_proc)

    # Print dataset shapes
    print_dataset_shape(train_tokenized_dataset, "Training Dataset")
    print_dataset_shape(validation_tokenized_dataset, "Validation Dataset")

    # Save tokenized datasets to disk
    train_tokenized_dataset.save_to_disk(os.path.join(args.output_dir, "train"))
    validation_tokenized_dataset.save_to_disk(os.path.join(args.output_dir, "validation"))
    test_tokenized_dataset.save_to_disk(os.path.join(args.output_dir, "test"))
    print("Tokenized datasets saved to disk.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, tokenize, and cache datasets for training.")
    parser.add_argument("--cache_dir", type=str, default="/workspace/ML_team/hf_cache", help="Path to Hugging Face cache directory.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer directory.")
    parser.add_argument("--dataset_name", type=str, default="DKYoon/SlimPajama-6B", help="Dataset name on Hugging Face hub.")
    parser.add_argument("--output_dir", type=str, default="./datasets_pack_full/tokenized_data", help="Directory to save tokenized datasets.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for tokenization.")

    args = parser.parse_args()
    prepare_and_cache_dataset(args)
