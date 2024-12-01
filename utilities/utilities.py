from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, Trainer
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
import os
import random
import numpy as np
import torch
import json
import time

CACHE_DIR = "./hf_cache"
SAMPLE_OUTPUT_FILE = os.path.join(CACHE_DIR, "tokenized_sample_output.txt")


def determine_compute_dtype_and_attention():
    """
    Determines the appropriate compute dtype and attention implementation
    based on the system's bfloat16 support.

    Returns:
        dict: A dictionary containing the compute dtype and attention implementation.
    """
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
    
    print("-------------------------")
    print(f"**compute_dtype={compute_dtype}")
    print(f"**attn_implementation={attn_implementation}")
    print("-------------------------")
    
    return {"compute_dtype": compute_dtype, "attn_implementation": attn_implementation}

def check_cuda_availability():
    """
    Checks if CUDA is available and prints the available CUDA devices.

    Returns:
        dict: A dictionary containing the number of CUDA devices and their names.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available. Device count: {device_count}")
        device_names = {}
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
            device_names[i] = device_name
        return {"available": True, "device_count": device_count, "devices": device_names}
    else:
        print("CUDA is not available. Running on CPU.")
        return {"available": False, "device_count": 0, "devices": {}}


def estimate_memory_usage(config_path='./configs/model_configs/llama_8b_config.json', num_gpus=4, num_nodes=1):
    """
    Estimates memory usage for DeepSpeed ZeRO Stage 2 and 3.

    Args:
        config_path (str): Path to the model configuration file.
        num_gpus (int): Number of GPUs per node.
        num_nodes (int): Number of nodes.

    Returns:
        dict: A dictionary containing memory estimates for ZeRO Stage 2 and ZeRO Stage 3.
    """
    try:
        # Load model configuration

        config = LlamaConfig.from_pretrained(config_path)
        model = LlamaForCausalLM(config=config)

        if hasattr(config, "_name_or_path"):
            print(f"Estimating memory usage for model: {config._name_or_path}")
        else:
            print("Model name not available. Proceeding with memory estimation.")
        
        # Estimate memory for ZeRO Stage 3
        zero3_mem_estimate = estimate_zero3_model_states_mem_needs_all_live(
            model, num_gpus_per_node=num_gpus, num_nodes=num_nodes
        )
        
        # Estimate memory for ZeRO Stage 2
        zero2_mem_estimate = estimate_zero2_model_states_mem_needs_all_live(
            model, num_gpus_per_node=num_gpus, num_nodes=num_nodes
        )
        
        return {
            "ZeRO Stage 3": zero3_mem_estimate,
            "ZeRO Stage 2": zero2_mem_estimate,
        }
    except Exception as e:
        raise RuntimeError(f"Error estimating memory usage: {e}")

def seed_everything(seed: int):
    """
    Sets random seed for reproducibility across different libraries.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_samples_to_json(samples, output_file):
    """
    Saves the given samples as a JSON file.

    Args:
        samples (list): A list of dataset samples to save.
        output_file (str): Path to the file where samples will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, "w") as f:
            json.dump(samples, f, indent=4)  # Save the dataset slice as JSON with indentation
        print(f"Samples saved successfully to: {output_file}")
    except Exception as e:
        print(f"Error saving samples to file: {e}")


def print_dataset_shape(dataset, dataset_name="Dataset", output_file="./hf_cache/tokenized_sample_output.json"):
    """
    Prints the shape and structure of a dataset and saves the first three samples
    to a file using the `save_samples_to_file` function.

    Args:
        dataset: The dataset to analyze.
        dataset_name (str): Name of the dataset for display.
        output_file (str): Path to the file where samples will be saved.
    """
    print(f"\n{dataset_name} Details:")
    print(f"  Total number of samples: {len(dataset)}")
    
    # Get the first three samples
    samples = dataset[:3]
    
    save_samples_to_json(samples, output_file)


def MFU_calculation(batch_size, sequence_length, model_name, number_of_GPU, GPU_peak_TFLOPS, iteration_time):
    """
    Calculates Model FLOPs Utilization (MFU) for a given model and hardware setup.

    Parameters:
    - batch_size (int): Batch size used in training.
    - sequence_length (int): Sequence length of the input data.
    - model_name (str): Name of the model (e.g., 'llama3-8b', 'llama3.2-1b', 'llama2-7b').
    - number_of_GPU (int): Number of GPUs used.
    - GPU_peak_TFLOPS (float): Theoretical peak TFLOPs of a single GPU.
    - iteration_time (float): Time taken for one iteration (in seconds).

    Returns:
    - str: MFU as a percentage rounded to 4 decimal places.
    """

    # Model-specific parameters based on the provided model_name
    model_params = {
        'llama3.1-8b': {
            'v': 128256,  # Vocabulary size
            'n': 32,       # Number of attention heads
            'h': 4096,    # Hidden state dimension
            'i': 14336,   # SwiGLU projection dimension
            'N': 32       # Number of layers
        },
        'llama3.2-1b': {
            'v': 128256,
            'n': 32,
            'h': 2048,
            'i': 8192,
            'N': 16
        },
        'llama2-7b': {
            'v': 32000,
            'n': 32,
            'h': 4096,
            'i': 11008,
            'N': 32
        }
    }

    # Ensure model_name is valid
    if model_name not in model_params:
        raise ValueError(f"Model '{model_name}' not recognized. Valid models: {list(model_params.keys())}")

    # Extract model-specific parameters
    params = model_params[model_name]
    v, n, h, i, N = params['v'], params['n'], params['h'], params['i'], params['N']

    b = batch_size
    s = sequence_length

    # FLOPs calculation for one forward pass
    flops_per_forward = (
        N * (6 * b * s * h**2 + 4 * b * s**2 * h + 3 * b * s**2 * n + 2 * b * s * h**2)
        + N * (6 * b * s * h * i)
        + 2 * b * s * h * v
    )

    # Forward-backward pass is roughly 3 times the forward pass FLOPs
    flops_per_forward_backward = (3 * flops_per_forward) / 10**12  # Convert to TFLOPs

    # GPU peak TFLOPs per iteration
    GPU_TFLOPs_per_iteration = number_of_GPU * GPU_peak_TFLOPS * iteration_time

    # Calculate MFU
    MFU = (flops_per_forward_backward / GPU_TFLOPs_per_iteration) * 100

    # Return MFU rounded to 4 decimal places
    return f"MFU: {MFU:.4f}%"


def infer_from_checkpoint(
    model_path: str = '/workspace/ML_team/train/model_checkpoints/checkpoint-2000',
    prompt: str = "Hello, how's it going?",
    max_length: int = 50
) -> str:
    """
    Perform inference using a pretrained language model checkpoint with a default device as CUDA.
    
    :param model_path: Path to the directory containing the model checkpoint.
    :param prompt: Input prompt for the model. Defaults to a generic prompt.
    :param max_length: Maximum length of the generated text.
    :return: Generated text.
    """
    try:
        # Default device to CUDA
        if torch.cuda.is_available():
            device = "cuda"
        else:
            raise RuntimeError("CUDA device is not available. Please ensure a GPU is accessible.")
        
        # Load the tokenizer and model from the checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Set the pad token if it does not exist
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings

        # Move the model to the CUDA device
        model.to(device)
        
        # Encode the input prompt and move it to CUDA
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        # Please corresponding attention masks
        # Generate text with specific settings
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,  # Handle pad tokens
            do_sample=True,  # Enable sampling for diverse outputs
            top_k=50,  # Top-k sampling
            top_p=0.9  # Nucleus sampling
        )
        
        # Decode and return the generated text
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"An error occurred during inference: {str(e)}"

def print_memory_usage(step, stage):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Step {step} ({stage}): Allocated: {allocated / (1024 ** 3):.2f} GB, Reserved: {reserved / (1024 ** 3):.2f}")

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start() before stop().")
        elapsed_time = time.time() - self.start_time
        self.start_time = None  # Reset timer
        return elapsed_time

def print_training_metrics(step, stage, duration=None):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    time_info = f"{duration:.2f} s" if duration else "N/A"
    print(f"Step {step} ({stage}): Allocated: {allocated / (1024 ** 3):.2f} GB, Reserved: {reserved / (1024 ** 3):.2f}, Duration: {time_info}")

class TrainerMemoryMonitor(Trainer):
    def log_training_metrics(self, step, stage, duration):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        time_info = f"{duration:.2f} s" if duration else "N/A"

        self.state.log_history.append({
            'step': step,
            'stage': stage,
            'allocated': f"{allocated / (1024 ** 3):.2f} GB",
            'reserved': f"{reserved / (1024 ** 3):.2f} GB",
            'duration': time_info
        })


    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step
        step_timer = Timer()
        step_timer.start()
        print_training_metrics(step, "training_step> before")

        forward_timer = Timer()
        forward_timer.start()

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            print_training_metrics(step, "forward pass: -before")
            loss = self.compute_loss(model, inputs)
            print_training_metrics(step, "forward pass: -after", forward_timer.stop())

        backward_timer = Timer()
        backward_timer.start()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        torch.cuda.empty_cache()
        print_training_metrics(step, "backward pass\ before")

        if self.use_apex: 
            with torch.cuda.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        print_training_metrics(step, "backward pass /after", backward_timer.stop())
        print_training_metrics(step, "training_step> after", step_timer.stop())

        torch.cuda.empty_cache()

        return loss.detach() / self.args.gradient_accumulation_steps
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inspect_model_params(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()}")

def calculate_layers(V, H, I, N, h=None, g_size=None):
    # Embedding Layer
    embedding_layer = V * H
    print(f"Embedding Layer: {embedding_layer}")

    # RMS Norm Layer
    rms_norm_layer = H
    print(f"RMS Norm Layer: {rms_norm_layer}")

    # Query (Q), Key (K), Value (V)
    q = H * H  # Or can be H * h * (H / h), but it simplifies to H * H
    if g_size is None:
        k = H * H
        v = H * H
    else:
        k = H * H / g_size
        v = H * H / g_size
    print(f"Query (Q): {q}")
    print(f"Key (K): {k}")
    print(f"Value (V): {v}")

    # Output (O)
    o = H * H
    print(f"Output (O): {o}")

    # MLP calculation
    mlp = 2 * H * I + I * H
    print(f"MLP: {mlp}")

    # Attention Blocks
    attention_blocks = N * (2 * rms_norm_layer + q + k + v + o + mlp)
    print(f"Attention Blocks: {attention_blocks}")

    # All Layers (Sum of all components)
    all_layers = embedding_layer + attention_blocks + rms_norm_layer 

    print(f"All Layers: {all_layers}")
    return all_layers


def calculate_optimal_scaling_law(C):
    """
    Calculate the optimal number of parameters (N), training tokens (D), 
    and expected loss (L) given a compute budget (C) based on Chinchilla scaling laws.

    Parameters:
    C (float): Compute budget in FLOPs.

    Returns:
    dict: A dictionary containing the optimal N, D, and L values.
    """
    # Calculate optimal N (number of parameters)
    N_opt = 0.6 * (C ** 0.45)
    
    # Calculate optimal D (number of training tokens)
    D_opt = 0.3 * (C ** 0.55)
    
    # Calculate optimal L (expected loss or optimal layers)
    L_opt = 1070 * (C ** -0.154) + 1.7
    
    result = {"N_opt": N_opt, "D_opt": D_opt, "L_opt": L_opt}
    print(result)
    return result


