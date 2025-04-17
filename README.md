# Phi-4-mini-instruct Benchmark Evaluation

This repository contains code for evaluating the Microsoft Phi-4-mini-instruct model on AIME24, AIME25, and GPQA benchmarks.

## Requirements

- Python 3.8+
- CUDA-capable GPU with 4x L4 GPUs (22GB each)
- Required Python packages (see requirements.txt)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases (optional but recommended for experiment tracking):
```bash
wandb login
```

## Running the Evaluation

To run the evaluation on all benchmarks:

```bash
python evaluate_phi4.py
```

This will:
1. Load the Phi-4-mini-instruct model
2. Evaluate on AIME24, AIME25, and GPQA benchmarks
3. Save results to JSON files
4. Log metrics to Weights & Biases (if configured)

## Model Configuration Fix

The code includes a fix for the RoPE (Rotary Position Embedding) scaling configuration issue in the Phi-4-mini-instruct model. The error occurs because the model's configuration expects a `short_factor` field with length 64, but the model provides a field with length 48.

### Solution: Monkey-patching the Validation

The code uses a monkey-patch approach to fix this issue:

1. Before loading the model configuration, we disable the strict RoPE scaling validation by replacing the validation method with a no-op function:
   ```python
   from transformers.models.phi3.configuration_phi3 import Phi3Config
   Phi3Config._rope_scaling_validation = lambda self: None
   ```

2. After loading the configuration, we adjust the length of the `short_factor` field to 64 by either:
   - Padding with zeros if the original length is less than 64
   - Truncating if the original length is greater than 64

This approach allows us to work around the validation issue without modifying the transformers library.

### Alternative Solutions

1. **Upgrade transformers**: If you're using an older version of the transformers library, upgrading to a newer version might fix this issue:
   ```bash
   pip install --upgrade transformers
   ```

2. **Use a different model**: If the issue persists, consider using a different model that doesn't have this configuration issue.

## Metrics

The evaluation includes the following metrics:
- Exact Match Accuracy
- Token Overlap Ratio
- Length Ratio

## Results

Results are saved in JSON format with the following structure:
```json
{
    "dataset": "benchmark_name",
    "results": [
        {
            "prompt": "input prompt",
            "reference": "reference answer",
            "generated": "model generated answer"
        },
        ...
    ],
    "timestamp": "evaluation timestamp"
}
```

## Notes

- The evaluation uses mixed precision (FP16) for efficient GPU memory usage
- The model is automatically distributed across available GPUs
- Generation parameters can be adjusted in the `generate_answer` method 