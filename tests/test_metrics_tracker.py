"""
Tests for MetricsTracker.

These tests verify the core functionality of metrics tracking:
- Recording screenshots, vision calls, orchestrator calls, and plan revisions
- Timing execution
- Producing comprehensive metrics summaries
- Calculating derived metrics like actions per screenshot
"""

import pytest
import time
from agi_agents.metrics_tracker import MetricsTracker
from agi_agents.models import MetricsSummary


def test_metrics_tracker_initialization():
    """Test that MetricsTracker initializes with zero counts."""
    tracker = MetricsTracker()
    
    assert tracker.screenshot_count == 0
    assert tracker.vision_calls == 0
    assert tracker.orchestrator_calls == 0
    assert tracker.plan_revisions == 0
    assert tracker.start_time is None
    assert tracker.end_time is None
    assert tracker.total_actions == 0
    assert tracker.successful_actions == 0


def test_record_screenshot():
    """Test recording screenshot captures."""
    tracker = MetricsTracker()
    
    tracker.record_screenshot()
    assert tracker.screenshot_count == 1
    
    tracker.record_screenshot()
    tracker.record_screenshot()
    assert tracker.screenshot_count == 3


def test_record_vision_call():
    """Test recording vision model invocations."""
    tracker = MetricsTracker()
    
    tracker.record_vision_call()
    assert tracker.vision_calls == 1
    
    tracker.record_vision_call()
    assert tracker.vision_calls == 2


def test_record_orchestrator_call():
    """Test recording orchestrator model invocations."""
    tracker = MetricsTracker()
    
    tracker.record_orchestrator_call()
    assert tracker.orchestrator_calls == 1
    
    tracker.record_orchestrator_call()
    tracker.record_orchestrator_call()
    assert tracker.orchestrator_calls == 3


def test_record_plan_revision():
    """Test recording plan revisions."""
    tracker = MetricsTracker()
    
    tracker.record_plan_revision()
    assert tracker.plan_revisions == 1
    
    tracker.record_plan_revision()
    assert tracker.plan_revisions == 2


def test_record_actions():
    """Test recording action statistics."""
    tracker = MetricsTracker()
    
    tracker.record_actions(total=5, successful=4)
    assert tracker.total_actions == 5
    assert tracker.successful_actions == 4
    
    tracker.record_actions(total=3, successful=3)
    assert tracker.total_actions == 8
    assert tracker.successful_actions == 7


def test_start_timing():
    """Test starting execution timing."""
    tracker = MetricsTracker()
    
    before = time.time()
    tracker.start()
    after = time.time()
    
    assert tracker.start_time is not None
    assert before <= tracker.start_time <= after


def test_stop_timing():
    """Test stopping execution timing."""
    tracker = MetricsTracker()
    tracker.start()
    
    before = time.time()
    tracker.stop()
    after = time.time()
    
    assert tracker.end_time is not None
    assert before <= tracker.end_time <= after
    assert tracker.end_time >= tracker.start_time


def test_get_summary_without_start_raises_error():
    """Test that get_summary raises error if start() was not called."""
    tracker = MetricsTracker()
    
    with pytest.raises(ValueError, match="Metrics tracking was not started"):
        tracker.get_summary()


def test_get_summary_basic():
    """Test getting basic metrics summary."""
    tracker = MetricsTracker()
    tracker.start()
    
    tracker.record_screenshot()
    tracker.record_screenshot()
    tracker.record_vision_call()
    tracker.record_vision_call()
    tracker.record_orchestrator_call()
    tracker.record_plan_revision()
    tracker.record_actions(total=5, successful=4)
    
    time.sleep(0.1)  # Ensure some time passes
    tracker.stop()
    
    summary = tracker.get_summary()
    
    assert isinstance(summary, MetricsSummary)
    assert summary.screenshot_count == 2
    assert summary.vision_calls == 2
    assert summary.orchestrator_calls == 1
    assert summary.plan_revisions == 1
    assert summary.total_actions == 5
    assert summary.successful_actions == 4
    assert summary.execution_time_seconds >= 0.1
    assert summary.actions_per_screenshot == 4 / 2  # 2.0


def test_get_summary_without_stop():
    """Test that get_summary works even if stop() was not called."""
    tracker = MetricsTracker()
    tracker.start()
    
    tracker.record_screenshot()
    tracker.record_actions(total=3, successful=3)
    
    time.sleep(0.05)
    
    summary = tracker.get_summary()
    
    assert summary.screenshot_count == 1
    assert summary.successful_actions == 3
    assert summary.execution_time_seconds >= 0.05


def test_get_summary_zero_screenshots():
    """Test that actions_per_screenshot is 0 when no screenshots captured."""
    tracker = MetricsTracker()
    tracker.start()
    
    tracker.record_actions(total=5, successful=5)
    tracker.stop()
    
    summary = tracker.get_summary()
    
    assert summary.screenshot_count == 0
    assert summary.actions_per_screenshot == 0.0


def test_get_summary_calculates_actions_per_screenshot():
    """Test that actions_per_screenshot is calculated correctly."""
    tracker = MetricsTracker()
    tracker.start()
    
    tracker.record_screenshot()
    tracker.record_screenshot()
    tracker.record_screenshot()
    tracker.record_actions(total=12, successful=9)
    tracker.stop()
    
    summary = tracker.get_summary()
    
    assert summary.actions_per_screenshot == 9 / 3  # 3.0


def test_metrics_tracker_full_workflow():
    """Test a complete metrics tracking workflow."""
    tracker = MetricsTracker()
    
    # Start tracking
    tracker.start()
    
    # Simulate task execution
    tracker.record_screenshot()  # Initial screenshot
    tracker.record_vision_call()  # Analyze screenshot
    tracker.record_orchestrator_call()  # Create plan
    
    tracker.record_actions(total=3, successful=3)  # Execute batch
    
    tracker.record_screenshot()  # Screenshot after navigation
    tracker.record_vision_call()  # Analyze new page
    tracker.record_orchestrator_call()  # Update plan
    
    tracker.record_actions(total=2, successful=1)  # Execute batch with error
    tracker.record_plan_revision()  # Revise plan due to error
    
    tracker.record_screenshot()  # Screenshot after error
    tracker.record_vision_call()  # Analyze page
    tracker.record_orchestrator_call()  # Revise plan
    
    tracker.record_actions(total=1, successful=1)  # Execute revised action
    
    # Stop tracking
    tracker.stop()
    
    # Get summary
    summary = tracker.get_summary()
    
    assert summary.screenshot_count == 3
    assert summary.vision_calls == 3
    assert summary.orchestrator_calls == 3
    assert summary.plan_revisions == 1
    assert summary.total_actions == 6
    assert summary.successful_actions == 5
    assert summary.execution_time_seconds > 0
    assert summary.actions_per_screenshot == 5 / 3  # ~1.67
