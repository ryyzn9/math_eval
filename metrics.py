from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoConfig
import torch
from tqdm import tqdm
import json

class MetricsCalculator:
    def __init__(self, model_name: str = "microsoft/phi-4-mini-instruct"):
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate various metrics for the evaluation results."""
        references = [r["reference"] for r in results]
        generated = [r["generated"] for r in results]
        
        # Tokenize for BLEU and other token-based metrics
        ref_tokens = [self.tokenizer.tokenize(ref) for ref in references]
        gen_tokens = [self.tokenizer.tokenize(gen) for gen in generated]
        
        # Calculate metrics
        metrics = {
            "exact_match": self._calculate_exact_match(references, generated),
            "token_overlap": self._calculate_token_overlap(ref_tokens, gen_tokens),
            "length_ratio": self._calculate_length_ratio(ref_tokens, gen_tokens)
        }
        
        return metrics
    
    def _calculate_exact_match(self, references: List[str], generated: List[str]) -> float:
        """Calculate exact match accuracy."""
        matches = sum(1 for ref, gen in zip(references, generated) if ref.strip() == gen.strip())
        return matches / len(references)
    
    def _calculate_token_overlap(self, ref_tokens: List[List[str]], gen_tokens: List[List[str]]) -> float:
        """Calculate token overlap ratio."""
        overlaps = []
        for ref, gen in zip(ref_tokens, gen_tokens):
            ref_set = set(ref)
            gen_set = set(gen)
            if len(ref_set) == 0:
                overlaps.append(0.0)
            else:
                overlap = len(ref_set.intersection(gen_set)) / len(ref_set)
                overlaps.append(overlap)
        return np.mean(overlaps)
    
    def _calculate_length_ratio(self, ref_tokens: List[List[str]], gen_tokens: List[List[str]]) -> float:
        """Calculate the ratio of generated length to reference length."""
        ratios = []
        for ref, gen in zip(ref_tokens, gen_tokens):
            if len(ref) == 0:
                ratios.append(1.0)
            else:
                ratio = len(gen) / len(ref)
                ratios.append(ratio)
        return np.mean(ratios)

def evaluate_results(results_file: str, metrics_calculator: MetricsCalculator) -> Dict[str, float]:
    """Evaluate results from a JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return metrics_calculator.calculate_metrics(results["results"]) 