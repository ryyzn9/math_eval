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