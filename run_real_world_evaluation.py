"""
Run real-world evaluation with reduced dataset sizes for faster testing.
"""

import sys
from test_real_world_evaluation import TestConfig, RealWorldEvaluator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_quick_evaluation():
    """Run a quick evaluation with smaller datasets."""
    # Create config with smaller sizes for testing
    config = TestConfig()
    config.small_corpus_size = 100
    config.medium_corpus_size = 500
    config.large_corpus_size = 1000
    config.query_batch_sizes = [1, 5, 10]
    config.concurrent_users = [1, 5, 10]
    config.test_duration_seconds = 30
    config.warmup_queries = 10
    
    logger.info("Starting quick real-world evaluation with reduced dataset sizes")
    logger.info(f"Corpus sizes: small={config.small_corpus_size}, medium={config.medium_corpus_size}, large={config.large_corpus_size}")
    
    evaluator = RealWorldEvaluator(config)
    
    try:
        evaluator.run_comprehensive_evaluation()
        logger.info("Quick evaluation completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run quick evaluation
    success = run_quick_evaluation()
    
    if success:
        print("\n" + "="*80)
        print("Quick evaluation completed! Check ./evaluation_results/ for detailed reports.")
        print("="*80)
        
        # Prompt for full evaluation
        response = input("\nRun full-scale evaluation? This may take 30-60 minutes. (y/n): ")
        if response.lower() == 'y':
            print("Starting full-scale evaluation...")
            from test_real_world_evaluation import main
            main()
    else:
        print("\nQuick evaluation failed. Please check the logs for errors.")
        sys.exit(1)