"""
AGI Agents package for adaptive web automation.

This package provides the core components for plan-based web automation
with dynamic error recovery using parallel vision and orchestrator models.
"""

from agi_agents.models import (
    # Vision Model Output Models
    PageElement,
    PageAnalysis,
    # Orchestrator Output Models
    Action,
    ExecutionPlan,
    # Error Handling Models
    ExecutionError,
    ExecutionResult,
    # Query Type Models
    ToolCall,
    InfoSeekingQuery,
    ElementLocation,
    ElementLocationFailure,
    # Performance Tracking Models
    MetricsSummary,
)
from agi_agents.metrics_tracker import MetricsTracker
from agi_agents.adaptive_agent import AdaptiveAgent

__all__ = [
    # Vision Model Output Models
    "PageElement",
    "PageAnalysis",
    # Orchestrator Output Models
    "Action",
    "ExecutionPlan",
    # Error Handling Models
    "ExecutionError",
    "ExecutionResult",
    # Query Type Models
    "ToolCall",
    "InfoSeekingQuery",
    "ElementLocation",
    "ElementLocationFailure",
    # Performance Tracking Models
    "MetricsSummary",
    "MetricsTracker",
    # Agent
    "AdaptiveAgent",
]
