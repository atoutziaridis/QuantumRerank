"""
Configuration utilities and tools for QuantumRerank system.

This module provides configuration file generation, comparison, migration,
backup/restore, and template generation utilities.
"""

import os
import shutil
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import hashlib

from .schemas import QuantumRerankConfigSchema, ValidationResult, validate_config_file
from .environments import get_config_for_environment, get_environment_defaults
from ..utils import get_logger


@dataclass
class ConfigDiff:
    """Configuration difference information."""
    added: Dict[str, Any]
    modified: Dict[str, Any]
    removed: Dict[str, Any]
    unchanged: Dict[str, Any]
    
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.added or self.modified or self.removed)
    
    def summary(self) -> str:
        """Get summary of changes."""
        changes = []
        if self.added:
            changes.append(f"{len(self.added)} added")
        if self.modified:
            changes.append(f"{len(self.modified)} modified")
        if self.removed:
            changes.append(f"{len(self.removed)} removed")
        
        if not changes:
            return "No changes"
        
        return ", ".join(changes)


class ConfigGenerator:
    """Configuration file generator with templates and best practices."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def generate_environment_config(
        self,
        environment: str,
        output_path: Union[str, Path],
        format: str = "yaml",
        include_comments: bool = True
    ) -> bool:
        """Generate configuration file for specific environment."""
        try:
            config = get_config_for_environment(environment)
            output_path = Path(output_path)
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "yaml":
                self._save_yaml_with_comments(config, output_path, include_comments)
            elif format.lower() == "json":
                config.to_json_file(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Generated {environment} configuration: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to generate configuration: {str(e)}")
            return False
    
    def _save_yaml_with_comments(
        self,
        config: QuantumRerankConfigSchema,
        output_path: Path,
        include_comments: bool
    ) -> None:
        """Save YAML configuration with explanatory comments."""
        config_dict = config.to_dict()
        
        if include_comments:
            yaml_content = self._generate_commented_yaml(config_dict)
            with open(output_path, 'w') as f:
                f.write(yaml_content)
        else:
            config.to_yaml_file(output_path)
    
    def _generate_commented_yaml(self, config_dict: Dict[str, Any]) -> str:
        """Generate YAML with explanatory comments."""
        lines = [
            "# QuantumRerank Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            "# This configuration follows PRD specifications and best practices",
            "",
            f"environment: {config_dict['environment']}  # Deployment environment",
            f"version: {config_dict['version']}  # Configuration version",
            "",
            "# Quantum computation settings (PRD constraints)",
            "quantum:",
            f"  n_qubits: {config_dict['quantum']['n_qubits']}  # PRD: 2-4 qubits maximum",
            f"  max_circuit_depth: {config_dict['quantum']['max_circuit_depth']}  # PRD: ≤15 gates",
            f"  shots: {config_dict['quantum']['shots']}  # Quantum measurement shots",
            f"  simulator_method: {config_dict['quantum']['simulator_method']}  # Quantum simulator",
            f"  enable_optimization: {str(config_dict['quantum']['enable_optimization']).lower()}  # Circuit optimization",
            "  quantum_backends:",
        ]
        
        for backend in config_dict['quantum']['quantum_backends']:
            lines.append(f"    - {backend}")
        
        lines.extend([
            "",
            "# Machine learning and embedding settings",
            "ml:",
            f"  embedding_model: {config_dict['ml']['embedding_model']}  # Recommended by docs",
            f"  embedding_dim: {config_dict['ml']['embedding_dim']}  # Embedding dimensions",
            f"  batch_size: {config_dict['ml']['batch_size']}  # PRD: 50-100 documents",
            f"  max_sequence_length: {config_dict['ml']['max_sequence_length']}  # Token limit",
            f"  use_quantum_compression: {str(config_dict['ml']['use_quantum_compression']).lower()}  # Enable compression",
            f"  compressed_dim: {config_dict['ml']['compressed_dim']}  # Compressed dimensions",
            "  parameter_prediction:",
            f"    hidden_dims: {config_dict['ml']['parameter_prediction']['hidden_dims']}",
            f"    dropout_rate: {config_dict['ml']['parameter_prediction']['dropout_rate']}",
            f"    learning_rate: {config_dict['ml']['parameter_prediction']['learning_rate']}",
            f"    activation: {config_dict['ml']['parameter_prediction']['activation']}",
            "",
            "# Performance targets (PRD specifications)",
            "performance:",
            f"  similarity_timeout_ms: {config_dict['performance']['similarity_timeout_ms']}  # PRD: <100ms",
            f"  batch_timeout_ms: {config_dict['performance']['batch_timeout_ms']}  # PRD: <500ms",
            f"  max_memory_gb: {config_dict['performance']['max_memory_gb']}  # PRD: <2GB",
            f"  cache_size: {config_dict['performance']['cache_size']}  # Cache entries",
            f"  enable_caching: {str(config_dict['performance']['enable_caching']).lower()}  # Enable caching",
            f"  max_concurrent_requests: {config_dict['performance']['max_concurrent_requests']}  # Concurrency limit",
            "",
            "# API service configuration",
            "api:",
            f"  host: {config_dict['api']['host']}  # Bind address",
            f"  port: {config_dict['api']['port']}  # Service port",
            f"  workers: {config_dict['api']['workers']}  # Worker processes",
            f"  rate_limit: {config_dict['api']['rate_limit']}  # Rate limiting",
            f"  enable_auth: {str(config_dict['api']['enable_auth']).lower()}  # Authentication",
            f"  cors_enabled: {str(config_dict['api']['cors_enabled']).lower()}  # CORS support",
            "  cors_origins:",
        ])
        
        for origin in config_dict['api']['cors_origins']:
            lines.append(f"    - {origin}")
        
        lines.extend([
            "",
            "# Monitoring and logging settings",
            "monitoring:",
            f"  log_level: {config_dict['monitoring']['log_level']}  # Logging verbosity",
            f"  enable_metrics: {str(config_dict['monitoring']['enable_metrics']).lower()}  # Metrics collection",
            f"  metrics_port: {config_dict['monitoring']['metrics_port']}  # Metrics endpoint",
            f"  health_check_interval_s: {config_dict['monitoring']['health_check_interval_s']}  # Health check frequency",
            f"  enable_tracing: {str(config_dict['monitoring']['enable_tracing']).lower()}  # Distributed tracing",
            f"  log_format: {config_dict['monitoring']['log_format']}  # Log format",
            f"  log_file: {config_dict['monitoring']['log_file']}  # Log file path",
            f"  max_log_file_size_mb: {config_dict['monitoring']['max_log_file_size_mb']}  # Log rotation",
            f"  log_retention_days: {config_dict['monitoring']['log_retention_days']}  # Log retention"
        ])
        
        return "\n".join(lines)
    
    def generate_all_environments(
        self,
        output_dir: Union[str, Path],
        format: str = "yaml"
    ) -> Dict[str, bool]:
        """Generate configuration files for all environments."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        environments = ["development", "testing", "staging", "production"]
        
        for env in environments:
            output_file = output_dir / f"{env}.{format}"
            results[env] = self.generate_environment_config(env, output_file, format)
        
        return results


class ConfigComparator:
    """Configuration comparison and diff utilities."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def compare_configs(
        self,
        config1: Union[QuantumRerankConfigSchema, Dict[str, Any], str, Path],
        config2: Union[QuantumRerankConfigSchema, Dict[str, Any], str, Path]
    ) -> ConfigDiff:
        """Compare two configurations and return differences."""
        dict1 = self._normalize_config(config1)
        dict2 = self._normalize_config(config2)
        
        added = {}
        modified = {}
        removed = {}
        unchanged = {}
        
        # Find added and modified keys
        for key, value in dict2.items():
            if key not in dict1:
                added[key] = value
            elif dict1[key] != value:
                if isinstance(value, dict) and isinstance(dict1[key], dict):
                    nested_diff = self.compare_configs(dict1[key], value)
                    if nested_diff.has_changes():
                        modified[key] = nested_diff
                    else:
                        unchanged[key] = value
                else:
                    modified[key] = {"old": dict1[key], "new": value}
            else:
                unchanged[key] = value
        
        # Find removed keys
        for key, value in dict1.items():
            if key not in dict2:
                removed[key] = value
        
        return ConfigDiff(
            added=added,
            modified=modified,
            removed=removed,
            unchanged=unchanged
        )
    
    def _normalize_config(
        self,
        config: Union[QuantumRerankConfigSchema, Dict[str, Any], str, Path]
    ) -> Dict[str, Any]:
        """Normalize configuration to dictionary format."""
        if isinstance(config, QuantumRerankConfigSchema):
            return config.to_dict()
        elif isinstance(config, dict):
            return config
        elif isinstance(config, (str, Path)):
            # Load from file
            config_path = Path(config)
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return QuantumRerankConfigSchema.from_yaml_file(config_path).to_dict()
            elif config_path.suffix.lower() == '.json':
                return QuantumRerankConfigSchema.from_json_file(config_path).to_dict()
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")


class ConfigMigrator:
    """Configuration migration and version upgrade utilities."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def migrate_to_environment(
        self,
        source_config: Union[str, Path],
        target_environment: str,
        output_path: Union[str, Path]
    ) -> bool:
        """Migrate configuration to different environment."""
        try:
            # Load source configuration
            source_path = Path(source_config)
            if source_path.suffix.lower() in ['.yaml', '.yml']:
                config = QuantumRerankConfigSchema.from_yaml_file(source_path)
            elif source_path.suffix.lower() == '.json':
                config = QuantumRerankConfigSchema.from_json_file(source_path)
            else:
                raise ValueError(f"Unsupported source config format: {source_path}")
            
            # Get target environment defaults
            target_config = get_config_for_environment(target_environment)
            
            # Apply source overrides to target base
            merged_config = self._merge_configs(target_config, config)
            merged_config.environment = target_environment
            
            # Save migrated configuration
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                merged_config.to_yaml_file(output_path)
            elif output_path.suffix.lower() == '.json':
                merged_config.to_json_file(output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_path}")
            
            self.logger.info(f"Configuration migrated to {target_environment}: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Configuration migration failed: {str(e)}")
            return False
    
    def _merge_configs(
        self,
        base_config: QuantumRerankConfigSchema,
        override_config: QuantumRerankConfigSchema
    ) -> QuantumRerankConfigSchema:
        """Merge two configurations with override precedence."""
        base_dict = base_config.to_dict()
        override_dict = override_config.to_dict()
        
        merged_dict = self._deep_merge(base_dict, override_dict)
        return QuantumRerankConfigSchema.from_dict(merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class ConfigBackup:
    """Configuration backup and restore utilities."""
    
    def __init__(self, backup_dir: Union[str, Path] = "config_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
    
    def create_backup(
        self,
        config_path: Union[str, Path],
        backup_name: Optional[str] = None
    ) -> Optional[Path]:
        """Create backup of configuration file."""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                self.logger.warning(f"Configuration file not found: {config_path}")
                return None
            
            # Generate backup name
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{config_path.stem}_{timestamp}{config_path.suffix}"
            
            backup_path = self.backup_dir / backup_name
            
            # Create backup
            shutil.copy2(config_path, backup_path)
            
            # Create metadata file
            metadata = {
                "original_path": str(config_path),
                "backup_time": datetime.now().isoformat(),
                "file_hash": self._calculate_file_hash(config_path)
            }
            
            metadata_path = backup_path.with_suffix(backup_path.suffix + ".meta")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Configuration backup created: {backup_path}")
            return backup_path
        
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            return None
    
    def restore_backup(
        self,
        backup_path: Union[str, Path],
        target_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """Restore configuration from backup."""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Load metadata if available
            metadata_path = backup_path.with_suffix(backup_path.suffix + ".meta")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if target_path is None:
                    target_path = Path(metadata["original_path"])
            
            if target_path is None:
                self.logger.error("Target path not specified and no metadata available")
                return False
            
            target_path = Path(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore file
            shutil.copy2(backup_path, target_path)
            
            self.logger.info(f"Configuration restored from backup: {target_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {str(e)}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available configuration backups."""
        backups = []
        
        for backup_file in self.backup_dir.glob("*"):
            if backup_file.suffix == ".meta":
                continue
            
            backup_info = {
                "path": str(backup_file),
                "name": backup_file.name,
                "size": backup_file.stat().st_size,
                "created": datetime.fromtimestamp(backup_file.stat().st_ctime)
            }
            
            # Load metadata if available
            metadata_path = backup_file.with_suffix(backup_file.suffix + ".meta")
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    backup_info.update(metadata)
                except Exception:
                    pass
            
            backups.append(backup_info)
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


def create_config_template(
    template_type: str = "basic",
    output_path: Union[str, Path] = "config_template.yaml"
) -> bool:
    """Create configuration template with explanations."""
    generator = ConfigGenerator()
    
    if template_type == "basic":
        return generator.generate_environment_config("development", output_path)
    elif template_type == "production":
        return generator.generate_environment_config("production", output_path)
    elif template_type == "all":
        output_dir = Path(output_path).parent / "templates"
        results = generator.generate_all_environments(output_dir)
        return all(results.values())
    else:
        raise ValueError(f"Unknown template type: {template_type}")


__all__ = [
    "ConfigDiff",
    "ConfigGenerator",
    "ConfigComparator",
    "ConfigMigrator",
    "ConfigBackup",
    "create_config_template"
]