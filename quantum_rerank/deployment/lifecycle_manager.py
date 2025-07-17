"""
Lifecycle Manager for Edge Deployment

Manages the complete lifecycle of edge-deployed quantum-inspired RAG systems,
including updates, rollbacks, health monitoring, and configuration management.
"""

import time
import json
import shutil
import subprocess
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RollbackStrategy(Enum):
    """Rollback strategies for failed deployments."""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    MANUAL = "manual"
    AUTOMATED = "automated"


class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class UpdateConfig:
    """Configuration for system updates."""
    update_strategy: str = "blue_green"
    health_check_timeout: int = 300
    rollback_strategy: RollbackStrategy = RollbackStrategy.AUTOMATED
    backup_enabled: bool = True
    validation_enabled: bool = True
    notification_enabled: bool = True
    max_rollback_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class DeploymentRecord:
    """Record of a deployment operation."""
    deployment_id: str
    timestamp: float
    version: str
    status: DeploymentStatus
    duration_seconds: float
    health_check_passed: bool
    rollback_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


class LifecycleManager:
    """Manages the complete lifecycle of edge deployments."""
    
    def __init__(self, config: UpdateConfig, deployment_path: Path):
        self.config = config
        self.deployment_path = deployment_path
        self.backup_path = deployment_path.parent / "backups"
        self.deployment_history: List[DeploymentRecord] = []
        self.current_version = "1.0.0"
        self.health_check_callbacks: List[Callable] = []
        
        # Ensure directories exist
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Lifecycle Manager initialized for {deployment_path}")
    
    def add_health_check(self, callback: Callable[[], bool]):
        """Add a health check callback."""
        self.health_check_callbacks.append(callback)
    
    def create_backup(self, version: str) -> Path:
        """Create backup of current deployment."""
        if not self.config.backup_enabled:
            return None
        
        backup_dir = self.backup_path / f"backup_{version}_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy deployment files
            if self.deployment_path.exists():
                shutil.copytree(self.deployment_path, backup_dir / "deployment", dirs_exist_ok=True)
            
            # Create backup metadata
            backup_metadata = {
                "version": version,
                "timestamp": time.time(),
                "backup_path": str(backup_dir),
                "deployment_path": str(self.deployment_path)
            }
            
            with open(backup_dir / "backup_metadata.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            logger.info(f"Backup created: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def perform_health_check(self) -> bool:
        """Perform comprehensive health check."""
        try:
            # Run all registered health checks
            for callback in self.health_check_callbacks:
                if not callback():
                    return False
            
            # Basic system health checks
            if not self._check_system_health():
                return False
            
            # Application-specific health checks
            if not self._check_application_health():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _check_system_health(self) -> bool:
        """Check system-level health."""
        try:
            # Check memory usage
            import psutil
            if psutil.virtual_memory().percent > 90:
                logger.warning("High memory usage detected")
                return False
            
            # Check disk space
            if psutil.disk_usage('/').percent > 95:
                logger.warning("Low disk space detected")
                return False
            
            # Check if deployment directory exists
            if not self.deployment_path.exists():
                logger.error("Deployment directory not found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return False
    
    def _check_application_health(self) -> bool:
        """Check application-specific health."""
        try:
            # Check if required files exist
            required_files = ["config.yaml", "models/", "scripts/"]
            for file_path in required_files:
                if not (self.deployment_path / file_path).exists():
                    logger.warning(f"Required file/directory missing: {file_path}")
            
            # Check if configuration is valid
            config_file = self.deployment_path / "config.yaml"
            if config_file.exists():
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if not isinstance(config, dict):
                        logger.error("Invalid configuration format")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Application health check failed: {e}")
            return False
    
    def deploy_update(self, new_version: str, update_package: Path) -> Dict[str, Any]:
        """Deploy system update."""
        deployment_id = f"deploy_{new_version}_{int(time.time())}"
        start_time = time.time()
        
        deployment_record = DeploymentRecord(
            deployment_id=deployment_id,
            timestamp=start_time,
            version=new_version,
            status=DeploymentStatus.DEPLOYING,
            duration_seconds=0,
            health_check_passed=False,
            rollback_count=0,
            metadata={"update_package": str(update_package)}
        )
        
        try:
            logger.info(f"Starting deployment: {deployment_id}")
            
            # Create backup of current version
            backup_path = self.create_backup(self.current_version)
            deployment_record.metadata["backup_path"] = str(backup_path) if backup_path else None
            
            # Deploy update based on strategy
            if self.config.update_strategy == "blue_green":
                success = self._deploy_blue_green(update_package, deployment_record)
            elif self.config.update_strategy == "rolling":
                success = self._deploy_rolling(update_package, deployment_record)
            else:
                success = self._deploy_direct(update_package, deployment_record)
            
            if success:
                # Perform health check
                health_check_passed = self.perform_health_check()
                deployment_record.health_check_passed = health_check_passed
                
                if health_check_passed:
                    deployment_record.status = DeploymentStatus.DEPLOYED
                    self.current_version = new_version
                    logger.info(f"Deployment successful: {deployment_id}")
                else:
                    # Health check failed, initiate rollback
                    logger.warning(f"Health check failed for {deployment_id}, initiating rollback")
                    rollback_success = self._perform_rollback(deployment_record, backup_path)
                    
                    if rollback_success:
                        deployment_record.status = DeploymentStatus.ROLLED_BACK
                    else:
                        deployment_record.status = DeploymentStatus.FAILED
            else:
                deployment_record.status = DeploymentStatus.FAILED
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployment_record.status = DeploymentStatus.FAILED
            deployment_record.metadata["error"] = str(e)
        
        finally:
            deployment_record.duration_seconds = time.time() - start_time
            self.deployment_history.append(deployment_record)
        
        return deployment_record.to_dict()
    
    def _deploy_blue_green(self, update_package: Path, deployment_record: DeploymentRecord) -> bool:
        """Deploy using blue-green strategy."""
        try:
            # Create green environment
            green_path = self.deployment_path.parent / "green_deployment"
            green_path.mkdir(parents=True, exist_ok=True)
            
            # Extract update package to green environment
            if update_package.suffix == '.tar.gz':
                subprocess.run(['tar', '-xzf', str(update_package), '-C', str(green_path)], 
                              check=True, capture_output=True)
            else:
                shutil.copytree(update_package, green_path / "deployment", dirs_exist_ok=True)
            
            # Validate green environment
            if not self._validate_deployment(green_path):
                return False
            
            # Switch to green (atomic operation)
            blue_path = self.deployment_path.parent / "blue_deployment"
            if self.deployment_path.exists():
                shutil.move(str(self.deployment_path), str(blue_path))
            
            shutil.move(str(green_path), str(self.deployment_path))
            
            # Clean up old blue environment
            if blue_path.exists():
                shutil.rmtree(blue_path)
            
            deployment_record.metadata["deployment_strategy"] = "blue_green"
            return True
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return False
    
    def _deploy_rolling(self, update_package: Path, deployment_record: DeploymentRecord) -> bool:
        """Deploy using rolling update strategy."""
        try:
            # For edge deployment, rolling update is similar to direct replacement
            # In a distributed system, this would update nodes one by one
            
            # Create temporary deployment
            temp_path = self.deployment_path.parent / "temp_deployment"
            temp_path.mkdir(parents=True, exist_ok=True)
            
            # Extract update package
            if update_package.suffix == '.tar.gz':
                subprocess.run(['tar', '-xzf', str(update_package), '-C', str(temp_path)], 
                              check=True, capture_output=True)
            else:
                shutil.copytree(update_package, temp_path / "deployment", dirs_exist_ok=True)
            
            # Validate deployment
            if not self._validate_deployment(temp_path):
                return False
            
            # Replace deployment atomically
            old_path = self.deployment_path.parent / "old_deployment"
            if self.deployment_path.exists():
                shutil.move(str(self.deployment_path), str(old_path))
            
            shutil.move(str(temp_path), str(self.deployment_path))
            
            # Clean up
            if old_path.exists():
                shutil.rmtree(old_path)
            
            deployment_record.metadata["deployment_strategy"] = "rolling"
            return True
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            return False
    
    def _deploy_direct(self, update_package: Path, deployment_record: DeploymentRecord) -> bool:
        """Deploy using direct replacement strategy."""
        try:
            # Remove existing deployment
            if self.deployment_path.exists():
                shutil.rmtree(self.deployment_path)
            
            # Create new deployment directory
            self.deployment_path.mkdir(parents=True, exist_ok=True)
            
            # Extract update package
            if update_package.suffix == '.tar.gz':
                subprocess.run(['tar', '-xzf', str(update_package), '-C', str(self.deployment_path)], 
                              check=True, capture_output=True)
            else:
                shutil.copytree(update_package, self.deployment_path / "deployment", dirs_exist_ok=True)
            
            # Validate deployment
            if not self._validate_deployment(self.deployment_path):
                return False
            
            deployment_record.metadata["deployment_strategy"] = "direct"
            return True
            
        except Exception as e:
            logger.error(f"Direct deployment failed: {e}")
            return False
    
    def _validate_deployment(self, deployment_path: Path) -> bool:
        """Validate deployment package."""
        if not self.config.validation_enabled:
            return True
        
        try:
            # Check if required files exist
            required_files = ["config.yaml", "scripts/", "models/"]
            for file_path in required_files:
                if not (deployment_path / file_path).exists():
                    logger.error(f"Required file/directory missing in deployment: {file_path}")
                    return False
            
            # Validate configuration
            config_file = deployment_path / "config.yaml"
            if config_file.exists():
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if not isinstance(config, dict):
                        logger.error("Invalid configuration format in deployment")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment validation failed: {e}")
            return False
    
    def _perform_rollback(self, deployment_record: DeploymentRecord, backup_path: Path) -> bool:
        """Perform rollback to previous version."""
        if not backup_path or not backup_path.exists():
            logger.error("No backup available for rollback")
            return False
        
        try:
            deployment_record.rollback_count += 1
            deployment_record.status = DeploymentStatus.ROLLING_BACK
            
            logger.info(f"Rolling back deployment: {deployment_record.deployment_id}")
            
            # Remove failed deployment
            if self.deployment_path.exists():
                shutil.rmtree(self.deployment_path)
            
            # Restore from backup
            backup_deployment = backup_path / "deployment"
            if backup_deployment.exists():
                shutil.copytree(backup_deployment, self.deployment_path)
            
            # Verify rollback
            if self.perform_health_check():
                logger.info("Rollback successful")
                return True
            else:
                logger.error("Rollback failed health check")
                return False
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "current_version": self.current_version,
            "deployment_path": str(self.deployment_path),
            "total_deployments": len(self.deployment_history),
            "last_deployment": self.deployment_history[-1].to_dict() if self.deployment_history else None,
            "health_status": "healthy" if self.perform_health_check() else "unhealthy",
            "backup_enabled": self.config.backup_enabled,
            "validation_enabled": self.config.validation_enabled
        }
    
    def get_deployment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return [record.to_dict() for record in self.deployment_history[-limit:]]
    
    def cleanup_old_backups(self, keep_count: int = 5):
        """Clean up old backup files."""
        try:
            backup_dirs = [d for d in self.backup_path.iterdir() if d.is_dir()]
            backup_dirs.sort(key=lambda x: x.stat().st_mtime)
            
            # Keep only the most recent backups
            if len(backup_dirs) > keep_count:
                for old_backup in backup_dirs[:-keep_count]:
                    shutil.rmtree(old_backup)
                    logger.info(f"Removed old backup: {old_backup}")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def export_deployment_logs(self, filepath: str):
        """Export deployment logs to file."""
        export_data = {
            "export_timestamp": time.time(),
            "current_version": self.current_version,
            "deployment_history": [record.to_dict() for record in self.deployment_history],
            "configuration": self.config.to_dict(),
            "deployment_path": str(self.deployment_path)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Deployment logs exported to {filepath}")


# Utility functions
def create_update_package(source_path: Path, package_path: Path) -> bool:
    """Create update package from source directory."""
    try:
        # Create tar.gz package
        subprocess.run(['tar', '-czf', str(package_path), '-C', str(source_path.parent), source_path.name], 
                      check=True, capture_output=True)
        
        logger.info(f"Update package created: {package_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create update package: {e}")
        return False


def validate_update_package(package_path: Path) -> bool:
    """Validate update package format and contents."""
    try:
        # Check if package exists
        if not package_path.exists():
            return False
        
        # Check package format
        if package_path.suffix != '.gz':
            return False
        
        # Test package extraction
        result = subprocess.run(['tar', '-tzf', str(package_path)], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            return False
        
        # Check for required files in package
        required_files = ['config.yaml', 'scripts/', 'models/']
        file_list = result.stdout.split('\n')
        
        for required_file in required_files:
            if not any(required_file in f for f in file_list):
                logger.warning(f"Required file missing in package: {required_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Package validation failed: {e}")
        return False