"""
Unit tests for benchmark metrics collection system.

Tests the performance metrics collection functionality for
Task 08: Performance Benchmarking Framework.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import numpy as np

from quantum_rerank.benchmarks.metrics import (
    LatencyTracker, MemoryTracker, ThroughputTracker, BenchmarkMetrics,
    LatencyMeasurement, MemoryMeasurement, ThroughputMeasurement
)


class TestLatencyMeasurement:
    """Test LatencyMeasurement dataclass."""
    
    def test_latency_measurement_creation(self):
        """Test LatencyMeasurement creation."""
        measurement = LatencyMeasurement(
            operation="test_op",
            duration_ms=42.5,
            timestamp=time.time(),
            success=True,
            metadata={"test": "data"}
        )
        
        assert measurement.operation == "test_op"
        assert measurement.duration_ms == 42.5
        assert measurement.success is True
        assert measurement.metadata["test"] == "data"


class TestMemoryMeasurement:
    """Test MemoryMeasurement dataclass."""
    
    def test_memory_measurement_creation(self):
        """Test MemoryMeasurement creation."""
        measurement = MemoryMeasurement(
            operation="memory_test",
            memory_mb=128.5,
            timestamp=time.time(),
            peak_memory_mb=150.0
        )
        
        assert measurement.operation == "memory_test"
        assert measurement.memory_mb == 128.5
        assert measurement.peak_memory_mb == 150.0


class TestThroughputMeasurement:
    """Test ThroughputMeasurement dataclass."""
    
    def test_throughput_measurement_creation(self):
        """Test ThroughputMeasurement creation."""
        measurement = ThroughputMeasurement(
            operation="throughput_test",
            operations_per_second=100.5,
            total_operations=1005,
            duration_s=10.0,
            timestamp=time.time()
        )
        
        assert measurement.operation == "throughput_test"
        assert measurement.operations_per_second == 100.5
        assert measurement.total_operations == 1005
        assert measurement.duration_s == 10.0


class TestLatencyTracker:
    """Test LatencyTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create LatencyTracker for testing."""
        return LatencyTracker(max_history=100)
    
    def test_tracker_initialization(self, tracker):
        """Test LatencyTracker initialization."""
        assert len(tracker.measurements) == 0
        assert len(tracker.active_operations) == 0
        assert tracker.max_history == 100
    
    def test_track_operation_context_manager_success(self, tracker):
        """Test successful operation tracking with context manager."""
        with tracker.track_operation("test_operation"):
            time.sleep(0.01)  # 10ms
        
        assert len(tracker.measurements) == 1
        measurement = tracker.measurements[0]
        assert measurement.operation == "test_operation"
        assert measurement.duration_ms > 5  # Should be at least 5ms
        assert measurement.success is True
    
    def test_track_operation_context_manager_failure(self, tracker):
        """Test failed operation tracking with context manager."""
        with pytest.raises(ValueError):
            with tracker.track_operation("failing_operation"):
                raise ValueError("Test error")
        
        assert len(tracker.measurements) == 1
        measurement = tracker.measurements[0]
        assert measurement.operation == "failing_operation"
        assert measurement.success is False
    
    def test_track_operation_with_metadata(self, tracker):
        """Test operation tracking with metadata."""
        metadata = {"batch_size": 10, "model": "test"}
        
        with tracker.track_operation("test_with_metadata", metadata=metadata):
            time.sleep(0.005)
        
        measurement = tracker.measurements[0]
        assert measurement.metadata == metadata
    
    def test_manual_operation_tracking(self, tracker):
        """Test manual start/stop operation tracking."""
        operation_id = tracker.start_operation("manual_test")
        
        assert operation_id in tracker.active_operations
        
        time.sleep(0.01)
        duration = tracker.stop_operation(operation_id, success=True)
        
        assert operation_id not in tracker.active_operations
        assert duration > 5  # Should be at least 5ms
        assert len(tracker.measurements) == 1
        
        measurement = tracker.measurements[0]
        assert measurement.operation == "manual_test"
        assert measurement.success is True
    
    def test_stop_nonexistent_operation(self, tracker):
        """Test stopping non-existent operation."""
        duration = tracker.stop_operation("nonexistent_id")
        assert duration == 0.0
    
    def test_get_statistics_no_data(self, tracker):
        """Test statistics with no measurements."""
        stats = tracker.get_statistics()
        
        assert stats["count"] == 0
        assert "error" in stats
    
    def test_get_statistics_with_data(self, tracker):
        """Test statistics calculation with data."""
        # Add multiple measurements
        durations = [10.0, 20.0, 30.0, 15.0, 25.0]
        
        for i, duration in enumerate(durations):
            measurement = LatencyMeasurement(
                operation="test_op",
                duration_ms=duration,
                timestamp=time.time(),
                success=True
            )
            tracker.measurements.append(measurement)
        
        stats = tracker.get_statistics("test_op")
        
        assert stats["count"] == 5
        assert stats["success_rate"] == 1.0
        assert stats["mean_ms"] == 20.0  # Mean of durations
        assert stats["median_ms"] == 20.0
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 30.0
        assert stats["p95_ms"] > stats["median_ms"]
    
    def test_get_statistics_with_failures(self, tracker):
        """Test statistics with some failed operations."""
        # Add mixed success/failure measurements
        measurements = [
            LatencyMeasurement("test_op", 10.0, time.time(), True),
            LatencyMeasurement("test_op", 20.0, time.time(), False),
            LatencyMeasurement("test_op", 15.0, time.time(), True)
        ]
        
        for measurement in measurements:
            tracker.measurements.append(measurement)
        
        stats = tracker.get_statistics("test_op")
        
        assert stats["count"] == 3
        assert stats["success_rate"] == 2/3  # 2 out of 3 successful
    
    def test_get_statistics_time_window(self, tracker):
        """Test statistics with time window filtering."""
        current_time = time.time()
        
        # Add measurements with different timestamps
        old_measurement = LatencyMeasurement("test_op", 10.0, current_time - 100, True)
        recent_measurement = LatencyMeasurement("test_op", 20.0, current_time - 1, True)
        
        tracker.measurements.extend([old_measurement, recent_measurement])
        
        # Get statistics for last 10 seconds
        stats = tracker.get_statistics("test_op", time_window_s=10.0)
        
        assert stats["count"] == 1  # Only recent measurement
        assert stats["mean_ms"] == 20.0
    
    def test_prd_compliance_similarity(self, tracker):
        """Test PRD compliance checking for similarity operations."""
        measurement = LatencyMeasurement("similarity_test", 75.0, time.time(), True)
        tracker.measurements.append(measurement)
        
        stats = tracker.get_statistics("similarity_test")
        
        assert stats["prd_compliant"] is True  # 75ms < 100ms target
    
    def test_prd_compliance_batch(self, tracker):
        """Test PRD compliance checking for batch operations."""
        measurement = LatencyMeasurement("batch_test", 600.0, time.time(), True)
        tracker.measurements.append(measurement)
        
        stats = tracker.get_statistics("batch_test")
        
        assert stats["prd_compliant"] is False  # 600ms > 500ms target
    
    def test_thread_safety(self, tracker):
        """Test thread safety of LatencyTracker."""
        def worker_function(worker_id):
            for i in range(10):
                with tracker.track_operation(f"worker_{worker_id}_op_{i}"):
                    time.sleep(0.001)  # 1ms
        
        # Start multiple threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker_function, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(tracker.measurements) == 50  # 5 workers × 10 operations each
        assert all(measurement.success for measurement in tracker.measurements)


class TestMemoryTracker:
    """Test MemoryTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create MemoryTracker for testing."""
        return MemoryTracker(max_history=100)
    
    def test_tracker_initialization(self, tracker):
        """Test MemoryTracker initialization."""
        assert len(tracker.measurements) == 0
        assert tracker.max_history == 100
    
    @patch('psutil.Process')
    def test_track_memory_context_manager(self, mock_process, tracker):
        """Test memory tracking with context manager."""
        # Mock memory info - simulate memory increase
        memory_values = [100, 120, 110]  # MB: baseline, peak, final
        mock_process.return_value.memory_info.side_effect = [
            Mock(rss=val * 1024 * 1024) for val in memory_values
        ]
        
        with tracker.track_memory("memory_test"):
            pass  # Simulate some work
        
        assert len(tracker.measurements) == 1
        measurement = tracker.measurements[0]
        assert measurement.operation == "memory_test"
        assert measurement.memory_mb == 10.0  # Final - baseline
        assert measurement.peak_memory_mb == 20.0  # Peak - baseline
    
    @patch('psutil.Process')
    def test_get_current_memory(self, mock_process, tracker):
        """Test current memory retrieval."""
        mock_process.return_value.memory_info.return_value.rss = 256 * 1024 * 1024  # 256MB
        
        current_memory = tracker.get_current_memory_mb()
        assert current_memory == 256.0
    
    def test_get_statistics_no_data(self, tracker):
        """Test statistics with no measurements."""
        stats = tracker.get_statistics()
        
        assert stats["count"] == 0
        assert "error" in stats
    
    def test_get_statistics_with_data(self, tracker):
        """Test memory statistics calculation."""
        measurements = [
            MemoryMeasurement("test_op", 10.0, time.time(), 15.0),
            MemoryMeasurement("test_op", 20.0, time.time(), 25.0),
            MemoryMeasurement("test_op", 5.0, time.time(), 8.0)
        ]
        
        for measurement in measurements:
            tracker.measurements.append(measurement)
        
        stats = tracker.get_statistics("test_op")
        
        assert stats["count"] == 3
        assert stats["mean_delta_mb"] == (10.0 + 20.0 + 5.0) / 3
        assert stats["max_delta_mb"] == 20.0
        assert stats["min_delta_mb"] == 5.0
        assert stats["mean_peak_mb"] == (15.0 + 25.0 + 8.0) / 3
        assert stats["max_peak_mb"] == 25.0
    
    def test_prd_compliance_memory(self, tracker):
        """Test PRD compliance for memory usage."""
        # Simulate 100 document processing with <2GB usage
        measurement = MemoryMeasurement("test_100_docs", 1500.0, time.time())  # 1.5GB
        tracker.measurements.append(measurement)
        
        stats = tracker.get_statistics("test_100_docs")
        
        assert stats["prd_compliant"] is True  # 1500MB < 2048MB target


class TestThroughputTracker:
    """Test ThroughputTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create ThroughputTracker for testing."""
        return ThroughputTracker(max_history=100)
    
    def test_tracker_initialization(self, tracker):
        """Test ThroughputTracker initialization."""
        assert len(tracker.measurements) == 0
        assert tracker.max_history == 100
    
    def test_measure_throughput_success(self, tracker):
        """Test throughput measurement with successful operations."""
        def fast_operation(data):
            time.sleep(0.001)  # 1ms per operation
            return "success"
        
        ops_per_second = tracker.measure_throughput(
            operation_name="fast_op",
            operation_func=fast_operation,
            operation_data="test_data",
            target_duration_s=0.1  # 100ms total
        )
        
        assert ops_per_second > 0
        assert len(tracker.measurements) == 1
        
        measurement = tracker.measurements[0]
        assert measurement.operation == "fast_op"
        assert measurement.operations_per_second == ops_per_second
        assert measurement.total_operations > 0
    
    def test_measure_throughput_with_failures(self, tracker):
        """Test throughput measurement with operation failures."""
        def failing_operation(data):
            raise RuntimeError("Operation failed")
        
        ops_per_second = tracker.measure_throughput(
            operation_name="failing_op",
            operation_func=failing_operation,
            operation_data="test_data",
            target_duration_s=0.05
        )
        
        assert ops_per_second == 0.0
        assert len(tracker.measurements) == 1
        
        measurement = tracker.measurements[0]
        assert measurement.total_operations == 0
    
    def test_get_statistics_with_data(self, tracker):
        """Test throughput statistics calculation."""
        measurements = [
            ThroughputMeasurement("test_op", 100.0, 1000, 10.0, time.time()),
            ThroughputMeasurement("test_op", 200.0, 2000, 10.0, time.time()),
            ThroughputMeasurement("test_op", 150.0, 1500, 10.0, time.time())
        ]
        
        for measurement in measurements:
            tracker.measurements.append(measurement)
        
        stats = tracker.get_statistics("test_op")
        
        assert stats["count"] == 3
        assert stats["mean_ops_per_second"] == (100.0 + 200.0 + 150.0) / 3
        assert stats["max_ops_per_second"] == 200.0
        assert stats["min_ops_per_second"] == 100.0
        assert stats["total_operations"] == 4500


class TestBenchmarkMetrics:
    """Test comprehensive BenchmarkMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create BenchmarkMetrics for testing."""
        return BenchmarkMetrics()
    
    def test_metrics_initialization(self, metrics):
        """Test BenchmarkMetrics initialization."""
        assert hasattr(metrics, 'latency')
        assert hasattr(metrics, 'memory')
        assert hasattr(metrics, 'throughput')
        assert isinstance(metrics.latency, LatencyTracker)
        assert isinstance(metrics.memory, MemoryTracker)
        assert isinstance(metrics.throughput, ThroughputTracker)
    
    @patch('psutil.Process')
    def test_track_complete_operation(self, mock_process, metrics):
        """Test complete operation tracking (latency + memory)."""
        # Mock memory usage
        mock_process.return_value.memory_info.side_effect = [
            Mock(rss=100 * 1024 * 1024),  # 100MB baseline
            Mock(rss=120 * 1024 * 1024),  # 120MB peak
            Mock(rss=110 * 1024 * 1024)   # 110MB final
        ]
        
        with metrics.track_complete_operation("complete_test"):
            time.sleep(0.01)  # 10ms
        
        # Check latency tracking
        assert len(metrics.latency.measurements) == 1
        latency_measurement = metrics.latency.measurements[0]
        assert latency_measurement.operation == "complete_test"
        assert latency_measurement.duration_ms > 5
        
        # Check memory tracking
        assert len(metrics.memory.measurements) == 1
        memory_measurement = metrics.memory.measurements[0]
        assert memory_measurement.operation == "complete_test"
        assert memory_measurement.memory_mb == 10.0  # Final - baseline
    
    def test_get_comprehensive_report_no_data(self, metrics):
        """Test comprehensive report with no data."""
        report = metrics.get_comprehensive_report()
        
        assert "latency_stats" in report
        assert "memory_stats" in report
        assert "throughput_stats" in report
        assert "current_memory_mb" in report
        assert "report_timestamp" in report
        
        # All stats should indicate no data
        assert report["latency_stats"]["count"] == 0
        assert report["memory_stats"]["count"] == 0
        assert report["throughput_stats"]["count"] == 0
    
    def test_get_comprehensive_report_with_data(self, metrics):
        """Test comprehensive report with data."""
        # Add some measurements
        metrics.latency.measurements.append(
            LatencyMeasurement("test_op", 50.0, time.time(), True)
        )
        metrics.memory.measurements.append(
            MemoryMeasurement("test_op", 100.0, time.time())
        )
        metrics.throughput.measurements.append(
            ThroughputMeasurement("test_op", 200.0, 2000, 10.0, time.time())
        )
        
        report = metrics.get_comprehensive_report("test_op")
        
        assert report["latency_stats"]["count"] == 1
        assert report["memory_stats"]["count"] == 1
        assert report["throughput_stats"]["count"] == 1
        assert report["latency_stats"]["mean_ms"] == 50.0
    
    def test_check_prd_compliance_all_pass(self, metrics):
        """Test PRD compliance check with all metrics passing."""
        # Add measurements that meet PRD targets
        metrics.latency.measurements.extend([
            LatencyMeasurement("similarity", 75.0, time.time(), True),
            LatencyMeasurement("batch", 400.0, time.time(), True)
        ])
        metrics.memory.measurements.append(
            MemoryMeasurement("memory", 1500.0, time.time())  # 1.5GB < 2GB
        )
        
        compliance = metrics.check_prd_compliance()
        
        assert compliance["similarity_under_100ms"] is True
        assert compliance["batch_under_500ms"] is True
        assert compliance["memory_under_2gb"] is True
        assert compliance["overall_compliant"] is True
    
    def test_check_prd_compliance_some_fail(self, metrics):
        """Test PRD compliance check with some metrics failing."""
        # Add measurements that exceed PRD targets
        metrics.latency.measurements.extend([
            LatencyMeasurement("similarity", 150.0, time.time(), True),  # > 100ms
            LatencyMeasurement("batch", 300.0, time.time(), True)       # < 500ms
        ])
        metrics.memory.measurements.append(
            MemoryMeasurement("memory", 3000.0, time.time())  # 3GB > 2GB
        )
        
        compliance = metrics.check_prd_compliance()
        
        assert compliance["similarity_under_100ms"] is False
        assert compliance["batch_under_500ms"] is True
        assert compliance["memory_under_2gb"] is False
        assert compliance["overall_compliant"] is False
    
    def test_reset_all_measurements(self, metrics):
        """Test resetting all measurements."""
        # Add some measurements
        metrics.latency.measurements.append(
            LatencyMeasurement("test", 50.0, time.time(), True)
        )
        metrics.memory.measurements.append(
            MemoryMeasurement("test", 100.0, time.time())
        )
        metrics.throughput.measurements.append(
            ThroughputMeasurement("test", 200.0, 2000, 10.0, time.time())
        )
        
        # Reset all
        metrics.reset_all_measurements()
        
        assert len(metrics.latency.measurements) == 0
        assert len(metrics.memory.measurements) == 0
        assert len(metrics.throughput.measurements) == 0