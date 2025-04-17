# --- disable the strict RoPE‐scaling check so we can patch it ourselves ---
try:
    from transformers.models.phi3.configuration_phi3 import Phi3Config
    # make _rope_scaling_validation a no‑op
    Phi3Config._rope_scaling_validation = lambda self: None
except ImportError:
    # if for some reason the class lives elsewhere, you can print(dir()) to find it
    pass
# ------------------------------------------------------------------------

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import wandb # type: ignore
from typing import List, Dict, Any
import json
from datetime import datetime

class Phi4Evaluator:
    def __init__(self, model_name: str = "microsoft/phi-4-mini-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load configuration with custom RoPE scaling
        config = AutoConfig.from_pretrained(model_name)
        # Fix the RoPE scaling configuration
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if 'short_factor' in config.rope_scaling:
                # Ensure short_factor has length 64
                if len(config.rope_scaling['short_factor']) != 64:
                    # Create a new short_factor with length 64
                    original_factor = config.rope_scaling['short_factor']
                    if len(original_factor) < 64:
                        # Pad with zeros if needed
                        new_factor = original_factor + [0.0] * (64 - len(original_factor))
                    else:
                        # Truncate if too long
                        new_factor = original_factor[:64]
                    config.rope_scaling['short_factor'] = new_factor
        
        # Load model with the fixed configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate_answer(self, prompt: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_benchmark(self, dataset_name: str, split: str = "test") -> Dict[str, Any]:
        dataset = load_dataset(dataset_name, split=split)
        results = []
        
        for item in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            prompt = item["prompt"]
            reference = item["reference"]
            
            generated = self.generate_answer(prompt)
            
            results.append({
                "prompt": prompt,
                "reference": reference,
                "generated": generated
            })
            
        return {
            "dataset": dataset_name,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

def main():
    # Initialize wandb for experiment tracking
    wandb.init(project="phi4-evaluation", name="benchmark-evaluation")
    
    evaluator = Phi4Evaluator()
    benchmarks = ["AIME24", "AIME25", "GPQA"]
    
    for benchmark in benchmarks:
        print(f"\nEvaluating {benchmark}...")
        results = evaluator.evaluate_benchmark(benchmark)
        
        # Save results
        output_file = f"results_{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Log to wandb
        wandb.log({
            f"{benchmark}_results": results
        })
    
    wandb.finish()

if __name__ == "__main__":
    main() 