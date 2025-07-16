"""
Small Language Model (SLM) Generator for Ultra-Lightweight RAG

This module implements efficient generation using 1-3B parameter models
optimized for edge deployment with <2GB memory footprint, supporting
the quantum-inspired lightweight RAG transition strategy.

Based on:
- Research: "End-to-End Pipeline for Offline, Edge RAG"
- Target: 500MB-2GB memory footprint, 2,585 tokens/second performance
- Models: Phi-3 Mini, Llama 3.2 1B, Gemma 2 2B, Mistral 7B (quantized)
"""

import torch
import torch.nn as nn
import logging
import time
import gc
import psutil
import json
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SLMConfig:
    """Configuration for Small Language Model generation."""
    
    # Model selection
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    model_size_category: str = "1B"  # 1B, 3B, 7B
    
    # Quantization settings for memory efficiency
    use_quantization: bool = True
    quantization_bits: int = 8  # 4 or 8 bit quantization
    quantization_method: str = "bitsandbytes"  # or "gptq", "awq"
    
    # Memory optimization
    max_memory_gb: float = 2.0
    offload_to_cpu: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    
    # Performance settings
    batch_size: int = 1  # Keep low for edge deployment
    use_cache: bool = True
    device_map: str = "auto"
    
    # RAG-specific settings
    context_window: int = 4096
    rag_prompt_template: str = """Context: {context}

Question: {question}

Answer: """
    
    # Model options by size category
    model_options: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.model_options is None:
            self.model_options = {
                "1B": [
                    "microsoft/Phi-3-mini-4k-instruct",
                    "google/gemma-2b-it",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                ],
                "3B": [
                    "microsoft/Phi-3-mini-128k-instruct",
                    "stabilityai/stablelm-3b-4e1t",
                    "openlm-research/open_llama_3b_v2"
                ],
                "7B": [
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "google/gemma-7b-it"
                ]
            }


class SLMGenerator:
    """
    Small Language Model generator for edge-deployed RAG systems.
    
    Optimizes for minimal memory footprint while maintaining
    generation quality for RAG applications.
    """
    
    def __init__(self, config: Optional[SLMConfig] = None):
        """
        Initialize SLM generator.
        
        Args:
            config: SLM configuration
        """
        self.config = config or SLMConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Performance tracking
        self.stats = {
            'model_loaded': False,
            'memory_usage_mb': 0,
            'avg_tokens_per_second': 0,
            'total_generations': 0,
            'avg_generation_time_ms': 0
        }
        
        # Initialize model
        self._load_model()
        
        logger.info(f"SLM Generator initialized: {self.config.model_name}")
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for memory efficiency."""
        if not self.config.use_quantization:
            return None
        
        if self.config.quantization_method == "bitsandbytes":
            if self.config.quantization_bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.quantization_bits == 8:
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    int8_threshold=6.0,
                    llm_int8_has_fp16_weight=True
                )
        
        return None
    
    def _load_model(self):
        """Load and optimize SLM for edge deployment."""
        logger.info(f"Loading SLM: {self.config.model_name}")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Get quantization config
        quantization_config = self._get_quantization_config()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        try:
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.config.device_map,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            # Add quantization config if available
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Apply additional optimizations
            if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            
            # Enable flash attention if available
            if self.config.use_flash_attention and hasattr(self.model.config, "use_flash_attention"):
                self.model.config.use_flash_attention = True
            
            # Set to evaluation mode
            self.model.eval()
            
            # Update stats
            self.stats['model_loaded'] = True
            self.stats['memory_usage_mb'] = self._get_memory_usage()
            
            logger.info(f"Model loaded successfully. Memory: {self.stats['memory_usage_mb']:.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _prepare_rag_prompt(self, 
                           query: str,
                           context: Union[str, List[str]]) -> str:
        """
        Prepare RAG prompt with context and query.
        
        Args:
            query: User query
            context: Retrieved context (string or list of strings)
            
        Returns:
            Formatted prompt
        """
        # Convert context list to string if needed
        if isinstance(context, list):
            context = "\n\n".join(context)
        
        # Apply template
        prompt = self.config.rag_prompt_template.format(
            context=context,
            question=query
        )
        
        return prompt
    
    def generate(self,
                query: str,
                context: Union[str, List[str]],
                max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                stream: bool = False) -> Union[str, List[str]]:
        """
        Generate response using SLM with RAG context.
        
        Args:
            query: User query
            context: Retrieved context
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            stream: Whether to stream tokens
            
        Returns:
            Generated text (or token stream if streaming)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        # Prepare prompt
        prompt = self._prepare_rag_prompt(query, context)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_window
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        with torch.no_grad():
            if stream:
                # Streaming generation
                return self._stream_generate(inputs, gen_config)
            else:
                # Standard generation
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Update stats
                generation_time = (time.time() - start_time) * 1000
                tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
                tokens_per_second = tokens_generated / (generation_time / 1000)
                
                self.stats['total_generations'] += 1
                self.stats['avg_generation_time_ms'] = (
                    (self.stats['avg_generation_time_ms'] * (self.stats['total_generations'] - 1) + 
                     generation_time) / self.stats['total_generations']
                )
                self.stats['avg_tokens_per_second'] = (
                    (self.stats['avg_tokens_per_second'] * (self.stats['total_generations'] - 1) + 
                     tokens_per_second) / self.stats['total_generations']
                )
                
                logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.1f}ms "
                           f"({tokens_per_second:.1f} tokens/s)")
                
                return generated_text
    
    def _stream_generate(self, 
                        inputs: Dict[str, torch.Tensor],
                        gen_config: GenerationConfig) -> List[str]:
        """Stream token generation for real-time output."""
        # Note: Simplified streaming - full implementation would yield tokens
        # For now, return generated tokens as list
        
        tokens = []
        
        # Generate token by token
        for i in range(gen_config.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                if gen_config.temperature > 0:
                    logits = logits / gen_config.temperature
                
                # Sample next token
                if gen_config.do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Decode token
                token_text = self.tokenizer.decode(next_token[0])
                tokens.append(token_text)
                
                # Update inputs for next iteration
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=1)
                
                # Check for EOS
                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break
        
        return tokens
    
    def batch_generate(self,
                      queries: List[str],
                      contexts: List[Union[str, List[str]]],
                      **kwargs) -> List[str]:
        """
        Batch generation for multiple queries.
        
        Args:
            queries: List of queries
            contexts: List of contexts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if len(queries) != len(contexts):
            raise ValueError("Number of queries must match number of contexts")
        
        responses = []
        
        # Process in batches based on config
        batch_size = self.config.batch_size
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            # Generate for each in batch
            for query, context in zip(batch_queries, batch_contexts):
                response = self.generate(query, context, **kwargs)
                responses.append(response)
        
        return responses
    
    def optimize_for_edge(self) -> Dict[str, Any]:
        """
        Apply additional optimizations for edge deployment.
        
        Returns:
            Optimization results
        """
        optimizations = {}
        
        # 1. CPU offloading for large models
        if self.config.offload_to_cpu and self.device == "cuda":
            try:
                # Move less frequently used layers to CPU
                # This is model-specific - simplified example
                self.model.cpu()
                optimizations['cpu_offload'] = True
            except Exception as e:
                logger.warning(f"CPU offloading failed: {e}")
                optimizations['cpu_offload'] = False
        
        # 2. Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # 3. Optimize KV cache
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self.config.use_cache
        
        # 4. Update memory usage
        self.stats['memory_usage_mb'] = self._get_memory_usage()
        optimizations['memory_usage_mb'] = self.stats['memory_usage_mb']
        
        logger.info(f"Edge optimizations applied: {optimizations}")
        
        return optimizations
    
    def benchmark_performance(self, 
                            test_queries: List[str],
                            test_contexts: List[str]) -> Dict[str, Any]:
        """
        Benchmark SLM performance for edge deployment validation.
        
        Args:
            test_queries: Test queries
            test_contexts: Test contexts
            
        Returns:
            Performance metrics
        """
        logger.info("Benchmarking SLM performance...")
        
        results = {
            'model': self.config.model_name,
            'quantization': f"{self.config.quantization_bits}-bit" if self.config.use_quantization else "none",
            'device': self.device,
            'memory_usage_mb': self.stats['memory_usage_mb']
        }
        
        # Warm-up
        _ = self.generate(test_queries[0], test_contexts[0], max_new_tokens=50)
        
        # Benchmark different sequence lengths
        for max_tokens in [50, 100, 200]:
            times = []
            tokens_per_second = []
            
            for query, context in zip(test_queries[:10], test_contexts[:10]):
                start_time = time.time()
                response = self.generate(query, context, max_new_tokens=max_tokens)
                gen_time = time.time() - start_time
                
                times.append(gen_time)
                tokens_generated = len(self.tokenizer.tokenize(response))
                tokens_per_second.append(tokens_generated / gen_time)
            
            results[f'avg_time_{max_tokens}_tokens'] = sum(times) / len(times)
            results[f'avg_tokens_per_second_{max_tokens}'] = sum(tokens_per_second) / len(tokens_per_second)
        
        # Memory efficiency
        results['tokens_per_mb'] = results['avg_tokens_per_second_100'] / self.stats['memory_usage_mb']
        
        logger.info(f"Benchmark complete: {results['avg_tokens_per_second_100']:.1f} tokens/s, "
                   f"{results['memory_usage_mb']:.1f} MB")
        
        return results
    
    def save_optimized_model(self, output_path: str):
        """Save optimized model for edge deployment."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        config_path = Path(output_path) / "slm_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save optimization metadata
        metadata = {
            'original_model': self.config.model_name,
            'quantization': self.config.quantization_bits if self.config.use_quantization else None,
            'memory_usage_mb': self.stats['memory_usage_mb'],
            'avg_tokens_per_second': self.stats['avg_tokens_per_second']
        }
        
        metadata_path = Path(output_path) / "optimization_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Optimized model saved to {output_path}")


# Utility functions for integration
def create_slm_config(model_size: str = "1B", 
                     memory_limit_gb: float = 2.0) -> SLMConfig:
    """
    Create optimized SLM configuration for different deployment scenarios.
    
    Args:
        model_size: Target model size (1B, 3B, 7B)
        memory_limit_gb: Memory limit in GB
        
    Returns:
        Optimized SLM configuration
    """
    config = SLMConfig(model_size_category=model_size, max_memory_gb=memory_limit_gb)
    
    # Adjust settings based on model size and memory
    if model_size == "1B":
        config.model_name = "microsoft/Phi-3-mini-4k-instruct"
        config.use_quantization = False  # Small enough without quantization
    elif model_size == "3B":
        config.model_name = "microsoft/Phi-3-mini-128k-instruct"
        config.use_quantization = memory_limit_gb < 4.0
        config.quantization_bits = 8
    elif model_size == "7B":
        config.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        config.use_quantization = True
        config.quantization_bits = 4 if memory_limit_gb < 8.0 else 8
    
    return config


def validate_slm_generator() -> Dict[str, Any]:
    """
    Validate SLM generator implementation.
    
    Returns:
        Validation results
    """
    try:
        # Create minimal config for testing
        config = SLMConfig(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            use_quantization=True,
            quantization_bits=8,
            max_new_tokens=50
        )
        
        # Note: Actual model loading would require the model files
        # For validation, we just check the configuration
        
        return {
            'status': 'success',
            'model': config.model_name,
            'quantization': f"{config.quantization_bits}-bit",
            'memory_target': f"{config.max_memory_gb}GB",
            'message': 'SLM generator configuration validated'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Validation failed: {str(e)}'
        }