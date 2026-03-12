# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM reinforcement learning project focused on fine-tuning Llama-3.1-8B-Instruct to bidirectionally convert between natural language descriptions and Mermaid.js diagram code. The project uses QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning on GPU-constrained environments.

## Development Setup

### Environment Management
- Python 3.12+ required (managed via `.python-version`)
- Uses `uv` as the package manager with dependencies defined in `pyproject.toml`
- Virtual environment is located in `.venv/`

### Key Dependencies
- **Core ML**: `torch`, `transformers`, `peft`, `trl`, `accelerate`
- **Quantization**: `bitsandbytes` for 4-bit model loading
- **Data**: `datasets` for loading the Mermaid training dataset
- **Monitoring**: `wandb` for experiment tracking
- **Environment**: `dotenv` for secure credential management

### Installation
```bash
# Install dependencies (assuming uv is available)
uv sync

# Or using pip with requirements.txt
pip install -r requirements.txt
```

## Core Architecture

### Training Pipeline (`post-train.py`)
The main training script implements a bidirectional fine-tuning approach:

1. **Dataset Loading**: Uses `Celiadraw/text-to-mermaid` dataset (first 2000 samples)
2. **Bidirectional Formatting**: Randomly trains on both directions:
   - Text → Mermaid.js code generation
   - Mermaid.js code → Text explanation
3. **QLoRA Setup**: 4-bit quantization with LoRA adapters (rank=16, targeting attention layers)
4. **Training Configuration**: 
   - Batch size: 2 per device with 8 gradient accumulation steps
   - Learning rate: 2e-4
   - Max sequence length: 1024 tokens
5. **Monitoring**: Custom perplexity logging via Weights & Biases

### Model Loading & Inference (`test.py`)
Demonstrates how to:
- Load the base Llama-3.1-8B model in 4-bit mode
- Apply trained LoRA adapters from `./llama-3-8b-custom-adapter`
- Generate Mermaid charts using the fine-tuned model
- Uses proper Llama-3 chat template formatting

### Configuration Management
- Credentials stored in `.env` file (HF_TOKEN, WANDB_API_KEY)
- Connection testing utility (`test_keys.py`) verifies access to:
  - Weights & Biases account
  - Hugging Face gated model repository

## Key Files

- `post-train.py`: Main training script with bidirectional fine-tuning
- `test.py`: Model inference example with trained adapters
- `test_keys.py`: Credential verification utility
- `main.py`: Basic hello world script
- `pyproject.toml`: Project configuration and dependencies
- `.env`: Environment variables (not committed)

## Training Workflow

1. **Prerequisites**: Valid HF token with Llama-3.1 access + W&B API key
2. **Verification**: Run `python test_keys.py` to verify credentials
3. **Training**: Execute `python post-train.py` (saves adapters to `./mermaid-llama-v1`)
4. **Testing**: Update adapter path in `test.py` and run for inference

## GPU Requirements

Designed for single GPU training with QLoRA to fit 8B parameter model in ~24GB VRAM. Uses:
- 4-bit quantization with nf4 format
- Double quantization enabled
- Gradient checkpointing
- Device auto-mapping

## Output Artifacts

- **Trained Adapters**: Saved to configurable directory (default: `./mermaid-llama-v1`)
- **Wandb Logs**: Training metrics and perplexity tracking
- **Model State**: LoRA adapters + tokenizer for deployment