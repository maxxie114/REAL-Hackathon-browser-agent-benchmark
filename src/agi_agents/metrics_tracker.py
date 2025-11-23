"""
MetricsTracker for tracking execution metrics in the adaptive agent system.

This module provides the MetricsTracker class which records key performance
indicators during task execution, including:
- Screenshot captures
- Vision model invocations
- Orchestrator model invocations
- Plan revisions
- Execution timing

The tracker produces a MetricsSummary at the end of execution for performance analysis.
"""

import time
from typing import Optional

from .models import MetricsSummary


class MetricsTracker:
    """
    Tracks execution metrics for performance analysis.
    
    Records key events during task execution and provides a comprehensive
    summary of performance metrics. Tracks timing, model invocations,
    and action execution statistics.
    
    Attributes:
        screenshot_count: Number of screenshots captured
        vision_calls: Number of vision model invocations
        orchestrator_calls: Number of orchestrator model invocations
        plan_revisions: Number of plan revisions
        start_time: Unix timestamp when tracking started
        end_time: Unix timestamp when tracking ended
        total_actions: Total number of actions across all plans
        successful_actions: Number of actions that executed successfully
    """
    
    def __init__(self):
        """Initialize the metrics tracker with zero counts."""
        self.screenshot_count: int = 0
        self.vision_calls: int = 0
        self.orchestrator_calls: int = 0
        self.plan_revisions: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_actions: int = 0
        self.successful_actions: int = 0
    
    def start(self) -> None:
        """
        Start timing the execution.
        
        Records the current timestamp as the start time for execution timing.
        Should be called at the beginning of task execution.
        """
        self.start_time = time.time()
    
    def stop(self) -> None:
        """
        Stop timing the execution.
        
        Records the current timestamp as the end time for execution timing.
        Should be called at the end of task execution.
        """
        self.end_time = time.time()
    
    def record_screenshot(self) -> None:
        """
        Record a screenshot capture event.
        
        Increments the screenshot counter. Should be called each time
        a screenshot is captured during task execution.
        """
        self.screenshot_count += 1
    
    def record_vision_call(self) -> None:
        """
        Record a vision model invocation.
        
        Increments the vision model call counter. Should be called each time
        the vision model is invoked for page analysis or element location.
        """
        self.vision_calls += 1
    
    def record_orchestrator_call(self) -> None:
        """
        Record an orchestrator model invocation.
        
        Increments the orchestrator model call counter. Should be called each time
        the orchestrator model is invoked for plan generation or revision.
        """
        self.orchestrator_calls += 1
    
    def record_plan_revision(self) -> None:
        """
        Record a plan revision event.
        
        Increments the plan revision counter. Should be called each time
        the execution plan is revised due to errors or failures.
        """
        self.plan_revisions += 1
    
    def record_actions(self, total: int, successful: int) -> None:
        """
        Record action execution statistics.
        
        Updates the total and successful action counters. Should be called
        after action batch execution to track action success rates.
        
        Args:
            total: Number of actions attempted
            successful: Number of actions that completed successfully
        """
        self.total_actions += total
        self.successful_actions += successful
    
    def get_summary(self) -> MetricsSummary:
        """
        Get a comprehensive metrics summary.
        
        Produces a MetricsSummary containing all tracked metrics and
        calculated statistics like execution time and actions per screenshot.
        
        Returns:
            MetricsSummary with all tracked metrics and calculated statistics
            
        Raises:
            ValueError: If start() was not called before get_summary()
        """
        if self.start_time is None:
            raise ValueError("Metrics tracking was not started. Call start() before get_summary().")
        
        # Calculate execution time
        end = self.end_time if self.end_time is not None else time.time()
        execution_time = end - self.start_time
        
        # Calculate actions per screenshot (avoid division by zero)
        actions_per_screenshot = (
            self.successful_actions / self.screenshot_count 
            if self.screenshot_count > 0 
            else 0.0
        )
        
        return MetricsSummary(
            screenshot_count=self.screenshot_count,
            vision_calls=self.vision_calls,
            orchestrator_calls=self.orchestrator_calls,
            plan_revisions=self.plan_revisions,
            total_actions=self.total_actions,
            successful_actions=self.successful_actions,
            execution_time_seconds=execution_time,
            actions_per_screenshot=actions_per_screenshot
        )
