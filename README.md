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

The code includes a fix for the RoPE (Rotary Position Embedding) scaling configuration issue in the Phi-4-mini-instruct model. The error occurs because the model's configuration expects a `short_factor` field with length 64, but the model provides a field with length 48. The code automatically fixes this by:

1. Loading the model configuration
2. Checking if the RoPE scaling configuration has a `short_factor` field
3. Adjusting the length of the `short_factor` field to 64 by either:
   - Padding with zeros if the original length is less than 64
   - Truncating if the original length is greater than 64

This fix is applied in both the `Phi4Evaluator` and `MetricsCalculator` classes.

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