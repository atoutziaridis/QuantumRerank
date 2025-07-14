#!/usr/bin/env python3
"""
QPMeL Training Script

Train a Quantum Polar Metric Learning model for improved semantic similarity.
This script implements the training approach from the QPMeL paper, training 
quantum circuit parameters to optimize semantic similarity using triplet loss.

Usage:
    python train_qpmel.py --dataset nfcorpus --epochs 50 --batch-size 16
    python train_qpmel.py --dataset synthetic --triplets 5000 
    python train_qpmel.py --dataset sentence-transformers --lr 0.001
"""

import argparse
import logging
import sys
import torch
import json
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_rerank.training.qpmel_trainer import QPMeLTrainer, QPMeLTrainingConfig
from quantum_rerank.training.triplet_generator import (
    TripletGenerator, TripletGeneratorConfig,
    load_nfcorpus_triplets, load_msmarco_triplets, 
    load_sentence_transformers_triplets, create_synthetic_triplets
)
from quantum_rerank.ml.qpmel_circuits import QPMeLConfig
from quantum_rerank.core.embeddings import EmbeddingProcessor
from quantum_rerank.utils.logging_config import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train QPMeL quantum reranker")
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['nfcorpus', 'msmarco', 'sentence-transformers', 'synthetic', 'custom'],
                       help='Dataset to use for training')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing dataset files')
    parser.add_argument('--triplets-file', type=str, default=None,
                       help='Custom triplets JSON file')
    parser.add_argument('--num-triplets', type=int, default=5000,
                       help='Number of synthetic triplets to generate')
    
    # Model arguments
    parser.add_argument('--n-qubits', type=int, default=4,
                       help='Number of qubits in quantum circuits')
    parser.add_argument('--n-layers', type=int, default=1,
                       help='Number of entangling layers')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256],
                       help='Hidden dimensions for MLP')
    parser.add_argument('--enable-qrc', action='store_true', default=True,
                       help='Enable Quantum Residual Correction')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Triplet loss margin')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    
    # Control arguments
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval for batches')
    
    # Output arguments
    parser.add_argument('--save-model', type=str, default='models/qpmel_trained.pt',
                       help='Path to save trained model')
    parser.add_argument('--save-triplets', type=str, default=None,
                       help='Path to save generated triplets')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained model after training')
    parser.add_argument('--eval-dataset', type=str, default='synthetic',
                       help='Dataset for evaluation')
    
    return parser.parse_args()

def load_dataset(args) -> List[Tuple[str, str, str]]:
    """Load training triplets based on arguments."""
    logger = logging.getLogger(__name__)
    
    if args.dataset == 'synthetic':
        logger.info(f"Generating {args.num_triplets} synthetic triplets")
        triplets = create_synthetic_triplets(args.num_triplets)
        
    elif args.dataset == 'nfcorpus':
        logger.info("Loading NFCorpus dataset")
        try:
            triplets = load_nfcorpus_triplets(args.data_dir)
        except Exception as e:
            logger.error(f"Failed to load NFCorpus: {e}")
            logger.info("Falling back to synthetic data")
            triplets = create_synthetic_triplets(args.num_triplets)
    
    elif args.dataset == 'msmarco':
        logger.info("Loading MS MARCO dataset")
        try:
            triplets = load_msmarco_triplets(args.data_dir)
        except Exception as e:
            logger.error(f"Failed to load MS MARCO: {e}")
            logger.info("Falling back to synthetic data")
            triplets = create_synthetic_triplets(args.num_triplets)
    
    elif args.dataset == 'sentence-transformers':
        logger.info("Loading SentenceTransformers AllNLI dataset")
        try:
            triplets = load_sentence_transformers_triplets()
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformers data: {e}")
            logger.info("Falling back to synthetic data")
            triplets = create_synthetic_triplets(args.num_triplets)
    
    elif args.dataset == 'custom':
        if not args.triplets_file:
            raise ValueError("Must specify --triplets-file for custom dataset")
        
        logger.info(f"Loading custom triplets from {args.triplets_file}")
        generator = TripletGenerator()
        triplets = generator.load_triplets(args.triplets_file)
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if not triplets:
        raise ValueError("No triplets loaded. Check dataset configuration.")
    
    logger.info(f"Loaded {len(triplets)} triplets")
    
    # Save triplets if requested
    if args.save_triplets:
        generator = TripletGenerator()
        generator.save_triplets(triplets, args.save_triplets)
    
    return triplets

def create_configs(args):
    """Create configuration objects from arguments."""
    
    # QPMeL circuit configuration
    qpmel_config = QPMeLConfig(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        enable_qrc=args.enable_qrc,
        entangling_gate="zz",
        max_circuit_depth=15
    )
    
    # Training configuration
    training_config = QPMeLTrainingConfig(
        hidden_dims=args.hidden_dims,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        triplet_margin=args.margin,
        qpmel_config=qpmel_config,
        early_stopping_patience=args.early_stopping,
        validation_split=args.val_split,
        save_best_model=True,
        log_interval=args.log_interval,
        validate_interval=5
    )
    
    return qpmel_config, training_config

def evaluate_model(trainer: QPMeLTrainer, eval_triplets: List[Tuple[str, str, str]]):
    """Evaluate the trained model."""
    logger = logging.getLogger(__name__)
    
    logger.info("Evaluating trained QPMeL model...")
    
    # Get trained reranker
    reranker = trainer.get_trained_reranker()
    
    # Test on sample triplets
    test_triplets = eval_triplets[:100]  # Test on first 100 triplets
    
    correct_rankings = 0
    total_tests = 0
    
    for anchor, positive, negative in test_triplets:
        try:
            # Rerank the positive and negative documents
            candidates = [positive, negative]
            results = reranker.rerank(anchor, candidates, method="quantum", top_k=2)
            
            # Check if positive document is ranked higher
            if len(results) >= 2:
                top_doc = results[0]["text"]
                if top_doc == positive:
                    correct_rankings += 1
                total_tests += 1
                
        except Exception as e:
            logger.warning(f"Evaluation failed for triplet: {e}")
            continue
    
    if total_tests > 0:
        accuracy = correct_rankings / total_tests
        logger.info(f"Evaluation Results:")
        logger.info(f"  Tested triplets: {total_tests}")
        logger.info(f"  Correct rankings: {correct_rankings}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        
        return {
            "accuracy": accuracy,
            "correct_rankings": correct_rankings,
            "total_tests": total_tests
        }
    else:
        logger.warning("No successful evaluations")
        return None

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    from quantum_rerank.utils.logging_config import LogConfig
    log_config = LogConfig(console_level=args.log_level)
    setup_logging(config=log_config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting QPMeL training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load dataset
        triplets = load_dataset(args)
        
        # Create configurations
        qpmel_config, training_config = create_configs(args)
        
        # Initialize trainer
        logger.info("Initializing QPMeL trainer...")
        embedding_processor = EmbeddingProcessor()
        trainer = QPMeLTrainer(
            config=training_config,
            embedding_processor=embedding_processor,
            device=device
        )
        
        # Log model information
        model_info = trainer.model.get_model_info()
        logger.info(f"Model architecture: {model_info}")
        
        # Train model
        logger.info("Starting training...")
        training_history = trainer.train(triplets, save_path=args.save_model)
        
        # Save training history
        history_path = Path(args.save_model).with_suffix('.history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
        # Evaluate if requested
        if args.evaluate:
            # Load evaluation dataset
            if args.eval_dataset == args.dataset:
                eval_triplets = triplets
            else:
                logger.info(f"Loading evaluation dataset: {args.eval_dataset}")
                old_dataset = args.dataset
                args.dataset = args.eval_dataset
                eval_triplets = load_dataset(args)
                args.dataset = old_dataset  # Restore
            
            eval_results = evaluate_model(trainer, eval_triplets)
            
            if eval_results:
                # Save evaluation results
                eval_path = Path(args.save_model).with_suffix('.eval.json')
                with open(eval_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                logger.info(f"Evaluation results saved to {eval_path}")
        
        logger.info("QPMeL training completed successfully!")
        
        # Print summary
        final_metrics = training_history[-1] if training_history else {}
        logger.info("Final Training Summary:")
        logger.info(f"  Final train loss: {final_metrics.get('train_loss', 'N/A')}")
        logger.info(f"  Final val loss: {final_metrics.get('val_loss', 'N/A')}")
        logger.info(f"  Best val loss: {trainer.best_val_loss}")
        logger.info(f"  Total epochs: {len(training_history)}")
        logger.info(f"  Model saved to: {args.save_model}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()