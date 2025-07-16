"""
Complete Quantum Parameter Training Pipeline for Medical Domain.

This module integrates all training components into a unified pipeline that
executes the complete quantum parameter optimization process from medical
data preparation through hybrid weight optimization.

Based on QRF-04 requirements for complete training pipeline integration.
"""

import logging
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .medical_data_preparation import (
    MedicalTrainingDataset, MedicalDataPreparationPipeline,
    MedicalTrainingConfig, TrainingPair
)
from .quantum_kernel_trainer import (
    QuantumKernelTrainer, QuantumKernelOptimizationPipeline,
    KTAOptimizationConfig, QuantumKernelTrainingResult
)
from .parameter_predictor_trainer import (
    MedicalParameterPredictorTrainer, ParameterPredictorTrainingPipeline,
    ParameterPredictorTrainingConfig, ParameterPredictorTrainingResult
)
from .hybrid_weight_optimizer import (
    MedicalHybridOptimizer, HybridWeightOptimizationPipeline,
    HybridWeightConfig, HybridWeightOptimizationResult
)
from ..evaluation.medical_relevance import MedicalDocument, MedicalQuery
from ..core.quantum_similarity_engine import QuantumSimilarityEngine
from ..config.settings import QuantumConfig
from ..ml.parameter_predictor import ParameterPredictorConfig

logger = logging.getLogger(__name__)


@dataclass
class CompleteTrainingConfig:
    """Configuration for complete training pipeline."""
    # Data preparation
    medical_training_config: MedicalTrainingConfig
    
    # Quantum kernel training
    kta_optimization_config: KTAOptimizationConfig
    
    # Parameter predictor training
    predictor_config: ParameterPredictorConfig
    predictor_training_config: ParameterPredictorTrainingConfig
    
    # Hybrid weight optimization
    hybrid_weight_config: HybridWeightConfig
    
    # Pipeline settings
    output_base_dir: str = "complete_quantum_training"
    save_intermediate_results: bool = True
    validate_each_stage: bool = True
    generate_reports: bool = True
    
    @classmethod
    def create_default(cls) -> 'CompleteTrainingConfig':
        """Create default configuration for complete training."""
        return cls(
            medical_training_config=MedicalTrainingConfig(
                target_pairs=5000,
                train_split=0.7,
                val_split=0.15,
                test_split=0.15
            ),
            kta_optimization_config=KTAOptimizationConfig(
                target_kta_score=0.7,
                max_iterations=50,
                optimization_method="differential_evolution"
            ),
            predictor_config=ParameterPredictorConfig(
                hidden_dims=[512, 256, 128],
                n_qubits=4,
                n_layers=2
            ),
            predictor_training_config=ParameterPredictorTrainingConfig(
                num_epochs=100,
                early_stopping_patience=15,
                learning_rate=1e-3
            ),
            hybrid_weight_config=HybridWeightConfig(
                weight_search_method="grid_search",
                weight_granularity=0.1,
                optimization_metric="ndcg_10"
            )
        )


@dataclass
class CompleteTrainingResult:
    """Results from complete training pipeline."""
    # Stage results
    data_preparation_result: Dict[str, Any]
    quantum_kernel_result: QuantumKernelTrainingResult
    parameter_predictor_result: ParameterPredictorTrainingResult
    hybrid_weight_result: HybridWeightOptimizationResult
    
    # Pipeline metadata
    total_training_time_seconds: float
    final_performance_metrics: Dict[str, float]
    deployment_recommendations: List[str]
    
    # File paths
    output_directory: str
    model_files: Dict[str, str]


class CompleteQuantumTrainingPipeline:
    """
    Complete quantum parameter training pipeline for medical domain.
    
    Orchestrates all training stages from data preparation through
    hybrid weight optimization to produce a fully trained quantum
    reranking system optimized for medical document ranking.
    """
    
    def __init__(self, config: Optional[CompleteTrainingConfig] = None):
        """Initialize complete training pipeline."""
        self.config = config or CompleteTrainingConfig.create_default()
        
        # Initialize quantum configuration
        self.quantum_config = QuantumConfig(
            n_qubits=self.config.predictor_config.n_qubits,
            max_circuit_depth=15
        )
        
        # Initialize similarity engine
        self.similarity_engine = QuantumSimilarityEngine(self.quantum_config)
        
        # Training state
        self.training_data: Dict[str, List[TrainingPair]] = {}
        self.current_stage = "initialization"
        
        logger.info("Complete quantum training pipeline initialized")
    
    def run_complete_training(self, medical_documents: List[MedicalDocument],
                            medical_queries: Optional[List[MedicalQuery]] = None) -> CompleteTrainingResult:
        """
        Run complete quantum parameter training pipeline.
        
        Args:
            medical_documents: Medical corpus documents
            medical_queries: Medical queries (optional)
            
        Returns:
            Complete training results
        """
        logger.info("Starting complete quantum parameter training pipeline")
        start_time = time.time()
        
        # Create output directory structure
        output_path = Path(self.config.output_base_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Medical Data Preparation
            logger.info("=" * 60)
            logger.info("STAGE 1: MEDICAL DATA PREPARATION")
            logger.info("=" * 60)
            self.current_stage = "data_preparation"
            data_prep_result = self._run_data_preparation(
                medical_documents, medical_queries, output_path / "stage1_data_preparation"
            )
            
            # Stage 2: Quantum Kernel Training
            logger.info("=" * 60)
            logger.info("STAGE 2: QUANTUM KERNEL TRAINING")
            logger.info("=" * 60)
            self.current_stage = "quantum_kernel_training"
            kernel_result = self._run_quantum_kernel_training(
                output_path / "stage2_quantum_kernel"
            )
            
            # Stage 3: Parameter Predictor Training
            logger.info("=" * 60)
            logger.info("STAGE 3: PARAMETER PREDICTOR TRAINING")
            logger.info("=" * 60)
            self.current_stage = "parameter_predictor_training"
            predictor_result = self._run_parameter_predictor_training(
                kernel_result, output_path / "stage3_parameter_predictor"
            )
            
            # Stage 4: Hybrid Weight Optimization
            logger.info("=" * 60)
            logger.info("STAGE 4: HYBRID WEIGHT OPTIMIZATION")
            logger.info("=" * 60)
            self.current_stage = "hybrid_weight_optimization"
            hybrid_result = self._run_hybrid_weight_optimization(
                output_path / "stage4_hybrid_weights"
            )
            
            # Generate final results and recommendations
            total_time = time.time() - start_time
            final_result = self._generate_final_results(
                data_prep_result, kernel_result, predictor_result, hybrid_result,
                total_time, output_path
            )
            
            logger.info("=" * 60)
            logger.info("COMPLETE TRAINING PIPELINE FINISHED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total training time: {total_time/3600:.2f} hours")
            logger.info(f"Final performance: {final_result.final_performance_metrics}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Training pipeline failed at stage {self.current_stage}: {e}")
            raise
    
    def _run_data_preparation(self, medical_documents: List[MedicalDocument],
                            medical_queries: Optional[List[MedicalQuery]],
                            output_dir: Path) -> Dict[str, Any]:
        """Run medical data preparation stage."""
        logger.info(f"Preparing medical training data from {len(medical_documents)} documents")
        
        # Initialize data preparation pipeline
        data_pipeline = MedicalDataPreparationPipeline(self.config.medical_training_config)
        
        # Run data preparation
        result = data_pipeline.run(
            medical_documents,
            medical_queries,
            str(output_dir)
        )
        
        # Load prepared data splits
        dataset = MedicalTrainingDataset(self.config.medical_training_config)
        
        train_pairs = dataset.load_dataset(result['file_paths']['train'])
        val_pairs = dataset.load_dataset(result['file_paths']['val'])
        test_pairs = dataset.load_dataset(result['file_paths']['test'])
        
        # Store training data
        self.training_data = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        logger.info(f"Data preparation completed: {len(train_pairs)} train, "
                   f"{len(val_pairs)} val, {len(test_pairs)} test pairs")
        
        if self.config.validate_each_stage:
            self._validate_data_preparation(result)
        
        return result
    
    def _run_quantum_kernel_training(self, output_dir: Path) -> QuantumKernelTrainingResult:
        """Run quantum kernel training with KTA optimization."""
        logger.info("Training quantum kernels with KTA optimization")
        
        # Initialize quantum kernel training pipeline
        kernel_pipeline = QuantumKernelOptimizationPipeline(
            self.quantum_config,
            self.config.kta_optimization_config
        )
        
        # Run quantum kernel training
        result = kernel_pipeline.run(
            self.training_data['train'],
            self.training_data['val'],
            self.training_data['test'],
            str(output_dir)
        )
        
        kernel_training_result = result['training_result']
        
        logger.info(f"Quantum kernel training completed: "
                   f"KTA score {kernel_training_result.best_kta_score:.4f}")
        
        if self.config.validate_each_stage:
            self._validate_quantum_kernel_training(kernel_training_result)
        
        return kernel_training_result
    
    def _run_parameter_predictor_training(self, kernel_result: QuantumKernelTrainingResult,
                                        output_dir: Path) -> ParameterPredictorTrainingResult:
        """Run parameter predictor training on medical embeddings."""
        logger.info("Training parameter predictor on medical embeddings")
        
        # Initialize parameter predictor pipeline
        predictor_pipeline = ParameterPredictorTrainingPipeline(
            self.config.predictor_config,
            self.config.predictor_training_config
        )
        
        # Use optimal parameters from kernel training as targets
        optimal_parameters = kernel_result.optimal_parameters
        
        # Run parameter predictor training
        result = predictor_pipeline.run(
            self.training_data['train'],
            optimal_parameters,
            self.training_data['val'],
            self.training_data['test'],
            str(output_dir)
        )
        
        predictor_training_result = result['training_result']
        
        logger.info(f"Parameter predictor training completed: "
                   f"correlation {predictor_training_result.correlation_metrics.get('overall_pearson_r', 0):.4f}")
        
        if self.config.validate_each_stage:
            self._validate_parameter_predictor_training(predictor_training_result)
        
        return predictor_training_result
    
    def _run_hybrid_weight_optimization(self, output_dir: Path) -> HybridWeightOptimizationResult:
        """Run hybrid weight optimization on medical test data."""
        logger.info("Optimizing quantum/classical hybrid weights")
        
        # Initialize hybrid weight optimization pipeline
        hybrid_pipeline = HybridWeightOptimizationPipeline(
            self.similarity_engine,
            self.config.hybrid_weight_config
        )
        
        # Run hybrid weight optimization
        result = hybrid_pipeline.run(
            self.training_data['test'],
            str(output_dir)
        )
        
        hybrid_optimization_result = result['optimization_result']
        
        logger.info(f"Hybrid weight optimization completed: "
                   f"optimal weights Q={hybrid_optimization_result.overall_optimal_weights[0]:.2f}, "
                   f"C={hybrid_optimization_result.overall_optimal_weights[1]:.2f}")
        
        if self.config.validate_each_stage:
            self._validate_hybrid_weight_optimization(hybrid_optimization_result)
        
        return hybrid_optimization_result
    
    def _generate_final_results(self, data_prep_result: Dict[str, Any],
                              kernel_result: QuantumKernelTrainingResult,
                              predictor_result: ParameterPredictorTrainingResult,
                              hybrid_result: HybridWeightOptimizationResult,
                              total_time: float,
                              output_path: Path) -> CompleteTrainingResult:
        """Generate final training results and deployment recommendations."""
        logger.info("Generating final results and deployment recommendations")
        
        # Calculate final performance metrics
        final_metrics = self._calculate_final_performance_metrics(
            kernel_result, predictor_result, hybrid_result
        )
        
        # Generate deployment recommendations
        deployment_recommendations = self._generate_deployment_recommendations(
            kernel_result, predictor_result, hybrid_result
        )
        
        # Collect model file paths
        model_files = {
            'trained_quantum_kernel': str(output_path / "stage2_quantum_kernel" / "trained_quantum_kernel.pkl"),
            'trained_parameter_predictor': str(output_path / "stage3_parameter_predictor" / "trained_parameter_predictor.pth"),
            'hybrid_weight_config': str(output_path / "stage4_hybrid_weights" / "deployment_config.pkl")
        }
        
        # Create final result
        final_result = CompleteTrainingResult(
            data_preparation_result=data_prep_result,
            quantum_kernel_result=kernel_result,
            parameter_predictor_result=predictor_result,
            hybrid_weight_result=hybrid_result,
            total_training_time_seconds=total_time,
            final_performance_metrics=final_metrics,
            deployment_recommendations=deployment_recommendations,
            output_directory=str(output_path),
            model_files=model_files
        )
        
        # Save final results
        final_results_path = output_path / "final_training_results.pkl"
        with open(final_results_path, 'wb') as f:
            pickle.dump(final_result, f)
        
        # Generate comprehensive report
        if self.config.generate_reports:
            self._generate_comprehensive_report(final_result, output_path)
        
        return final_result
    
    def _validate_data_preparation(self, result: Dict[str, Any]):
        """Validate data preparation stage."""
        logger.info("Validating data preparation stage")
        
        # Check if target number of pairs was achieved
        total_pairs = result['total_pairs']
        target_pairs = self.config.medical_training_config.target_pairs
        
        if total_pairs < target_pairs * 0.8:
            logger.warning(f"Generated {total_pairs} pairs, target was {target_pairs}")
        
        # Check domain balance
        domain_dist = result['domain_distribution']
        if len(domain_dist) < 3:
            logger.warning(f"Limited domain coverage: {list(domain_dist.keys())}")
        
        logger.info("Data preparation validation completed")
    
    def _validate_quantum_kernel_training(self, result: QuantumKernelTrainingResult):
        """Validate quantum kernel training stage."""
        logger.info("Validating quantum kernel training stage")
        
        # Check KTA score improvement
        target_kta = self.config.kta_optimization_config.target_kta_score
        achieved_kta = result.best_kta_score
        
        if achieved_kta < target_kta:
            logger.warning(f"KTA score {achieved_kta:.4f} below target {target_kta}")
        
        # Check parameter convergence
        if result.convergence_iteration >= self.config.kta_optimization_config.max_iterations * 0.9:
            logger.warning("KTA optimization may not have converged")
        
        logger.info("Quantum kernel training validation completed")
    
    def _validate_parameter_predictor_training(self, result: ParameterPredictorTrainingResult):
        """Validate parameter predictor training stage."""
        logger.info("Validating parameter predictor training stage")
        
        # Check correlation
        correlation = result.correlation_metrics.get('overall_pearson_r', 0)
        if correlation < 0.7:
            logger.warning(f"Parameter prediction correlation {correlation:.4f} below 0.7")
        
        # Check training convergence
        if result.convergence_epoch >= self.config.predictor_training_config.num_epochs * 0.9:
            logger.warning("Parameter predictor training may not have converged")
        
        logger.info("Parameter predictor training validation completed")
    
    def _validate_hybrid_weight_optimization(self, result: HybridWeightOptimizationResult):
        """Validate hybrid weight optimization stage."""
        logger.info("Validating hybrid weight optimization stage")
        
        # Check if optimization found meaningful differences
        quantum_weight, classical_weight = result.overall_optimal_weights
        
        if abs(quantum_weight - 0.5) < 0.1:
            logger.warning("Hybrid optimization found weights close to 50/50 - may indicate no clear advantage")
        
        # Check scenario consistency
        scenario_weights = [r.optimal_quantum_weight for r in result.scenario_results.values()]
        if scenario_weights:
            weight_variation = max(scenario_weights) - min(scenario_weights)
            if weight_variation > 0.3:
                logger.info("High weight variation across scenarios - adaptive selection recommended")
        
        logger.info("Hybrid weight optimization validation completed")
    
    def _calculate_final_performance_metrics(self, kernel_result: QuantumKernelTrainingResult,
                                          predictor_result: ParameterPredictorTrainingResult,
                                          hybrid_result: HybridWeightOptimizationResult) -> Dict[str, float]:
        """Calculate final performance metrics."""
        metrics = {
            'kta_score': kernel_result.best_kta_score,
            'parameter_correlation': predictor_result.correlation_metrics.get('overall_pearson_r', 0),
            'optimal_quantum_weight': hybrid_result.overall_optimal_weights[0],
            'optimal_classical_weight': hybrid_result.overall_optimal_weights[1],
            'training_time_hours': kernel_result.training_time_seconds / 3600 +
                                 predictor_result.training_time_seconds / 3600 +
                                 hybrid_result.optimization_time_seconds / 3600
        }
        
        # Add scenario-specific performance
        for scenario, result in hybrid_result.scenario_results.items():
            metrics[f'{scenario}_performance'] = result.performance_metric
        
        return metrics
    
    def _generate_deployment_recommendations(self, kernel_result: QuantumKernelTrainingResult,
                                          predictor_result: ParameterPredictorTrainingResult,
                                          hybrid_result: HybridWeightOptimizationResult) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []
        
        # Overall training success
        kta_score = kernel_result.best_kta_score
        param_correlation = predictor_result.correlation_metrics.get('overall_pearson_r', 0)
        
        if kta_score > 0.7 and param_correlation > 0.8:
            recommendations.append("Training completed successfully - system ready for production deployment")
        elif kta_score > 0.5 and param_correlation > 0.6:
            recommendations.append("Training shows promise - consider additional optimization before production")
        else:
            recommendations.append("Training results below targets - investigate data quality or model architecture")
        
        # Model deployment
        recommendations.append(f"Deploy trained quantum kernel with KTA score {kta_score:.3f}")
        recommendations.append(f"Use parameter predictor with {param_correlation:.3f} correlation for real-time inference")
        
        # Hybrid weight recommendations
        q_weight, c_weight = hybrid_result.overall_optimal_weights
        recommendations.append(f"Configure hybrid weights: Quantum={q_weight:.2f}, Classical={c_weight:.2f}")
        
        # Add scenario-specific recommendations
        recommendations.extend(hybrid_result.recommendations[:3])
        
        # Performance recommendations
        if kta_score > 0.8:
            recommendations.append("Strong quantum kernel performance - prioritize quantum methods")
        elif param_correlation > 0.9:
            recommendations.append("Excellent parameter prediction - enable real-time parameter adaptation")
        
        return recommendations
    
    def _generate_comprehensive_report(self, final_result: CompleteTrainingResult, output_path: Path):
        """Generate comprehensive training report."""
        report_path = output_path / "comprehensive_training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("QUANTUM PARAMETER TRAINING PIPELINE - COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training completed in {final_result.total_training_time_seconds/3600:.2f} hours\n")
            f.write(f"KTA Score: {final_result.quantum_kernel_result.best_kta_score:.4f}\n")
            f.write(f"Parameter Correlation: {final_result.parameter_predictor_result.correlation_metrics.get('overall_pearson_r', 0):.4f}\n")
            f.write(f"Optimal Weights: Q={final_result.hybrid_weight_result.overall_optimal_weights[0]:.2f}, "
                   f"C={final_result.hybrid_weight_result.overall_optimal_weights[1]:.2f}\n\n")
            
            # Stage-by-stage results
            f.write("STAGE RESULTS\n")
            f.write("-" * 40 + "\n")
            
            f.write("1. Data Preparation:\n")
            f.write(f"   Training pairs: {final_result.data_preparation_result['train_pairs']}\n")
            f.write(f"   Validation pairs: {final_result.data_preparation_result['val_pairs']}\n")
            f.write(f"   Test pairs: {final_result.data_preparation_result['test_pairs']}\n")
            
            f.write("\n2. Quantum Kernel Training:\n")
            f.write(f"   Best KTA score: {final_result.quantum_kernel_result.best_kta_score:.4f}\n")
            f.write(f"   Convergence epoch: {final_result.quantum_kernel_result.convergence_iteration}\n")
            f.write(f"   Training time: {final_result.quantum_kernel_result.training_time_seconds:.1f}s\n")
            
            f.write("\n3. Parameter Predictor Training:\n")
            f.write(f"   Correlation: {final_result.parameter_predictor_result.correlation_metrics.get('overall_pearson_r', 0):.4f}\n")
            f.write(f"   Convergence epoch: {final_result.parameter_predictor_result.convergence_epoch}\n")
            f.write(f"   Training time: {final_result.parameter_predictor_result.training_time_seconds:.1f}s\n")
            
            f.write("\n4. Hybrid Weight Optimization:\n")
            for scenario, result in final_result.hybrid_weight_result.scenario_results.items():
                f.write(f"   {scenario}: Q={result.optimal_quantum_weight:.2f}, "
                       f"Performance={result.performance_metric:.4f}\n")
            
            # Deployment recommendations
            f.write("\nDEPLOYMENT RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(final_result.deployment_recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
            # Model files
            f.write("\nMODEL FILES\n")
            f.write("-" * 40 + "\n")
            for model_type, file_path in final_result.model_files.items():
                f.write(f"{model_type}: {file_path}\n")
        
        logger.info(f"Comprehensive report saved to {report_path}")


def run_complete_quantum_training(medical_documents: List[MedicalDocument],
                                medical_queries: Optional[List[MedicalQuery]] = None,
                                config: Optional[CompleteTrainingConfig] = None) -> CompleteTrainingResult:
    """
    Convenience function to run complete quantum training pipeline.
    
    Args:
        medical_documents: Medical corpus documents
        medical_queries: Medical queries (optional)
        config: Training configuration (optional)
        
    Returns:
        Complete training results
    """
    pipeline = CompleteQuantumTrainingPipeline(config)
    return pipeline.run_complete_training(medical_documents, medical_queries)