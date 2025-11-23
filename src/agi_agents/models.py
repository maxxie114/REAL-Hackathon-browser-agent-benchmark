"""
Data models for the adaptive web automation agent system.

This module defines all core data structures used in the plan-based execution
model with dynamic error recovery, including:
- Vision model outputs (PageElement, PageAnalysis)
- Orchestrator outputs (Action, ExecutionPlan)
- Error handling (ExecutionError, ExecutionResult)
- Query types (ToolCall, InfoSeekingQuery, ElementLocation, ElementLocationFailure)
- Performance tracking (MetricsSummary)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


# ============================================================================
# Vision Model Output Models
# ============================================================================

@dataclass
class PageElement:
    """
    Interactive element on a web page identified by the vision model.
    
    Attributes:
        element_id: Unique identifier for the element
        element_type: Type of element (e.g., "button", "input", "select", "link")
        label: Visible text or aria-label for the element
        coordinates: (x, y) center coordinates of the element
        field_type: For input elements, the specific type (e.g., "text", "email", "password")
        attributes: Additional HTML attributes for the element
    """
    element_id: str
    element_type: str
    label: str
    coordinates: Tuple[int, int]
    field_type: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class PageAnalysis:
    """
    Complete page analysis from the vision model.
    
    Contains all interactive elements identified on the page along with
    metadata about the page type and content.
    
    Attributes:
        elements: List of all interactive elements found on the page
        page_type: Classification of page (e.g., "form", "search", "content", "navigation")
        content_summary: Brief summary of the page content
        timestamp: Unix timestamp when the analysis was performed
    """
    elements: List[PageElement]
    page_type: str
    content_summary: str
    timestamp: float


# ============================================================================
# Orchestrator Output Models
# ============================================================================

@dataclass
class Action:
    """
    Single action in the execution plan.
    
    Represents a discrete step that the agent should take, with all
    necessary information to execute it.
    
    Attributes:
        action_type: Type of action (e.g., "click", "type", "select", "scroll", "wait")
        target_element_id: ID of the target element from PageAnalysis
        parameters: Action-specific parameters (e.g., text to type, option to select)
        is_navigation: Whether this action is expected to cause page navigation
        description: Human-readable description of what this action does
    """
    action_type: str
    target_element_id: str
    parameters: Dict[str, Any]
    is_navigation: bool
    description: str


@dataclass
class ExecutionPlan:
    """
    Dynamic execution plan maintained by the orchestrator.
    
    Contains the sequence of actions to execute along with the orchestrator's
    reasoning and revision history.
    
    Attributes:
        actions: Ordered list of actions to execute
        reasoning: Orchestrator's explanation for this plan
        created_at: Unix timestamp when the plan was created
        revision_count: Number of times this plan has been revised
    """
    actions: List[Action]
    reasoning: str = ""
    created_at: float = 0.0
    revision_count: int = 0


# ============================================================================
# Error Handling Models
# ============================================================================

@dataclass
class ExecutionError:
    """
    Error information from a failed action execution.
    
    Captures all relevant details about what went wrong during action
    execution to enable intelligent plan revision.
    
    Attributes:
        action: The action that failed
        error_type: Classification of error (e.g., "element_not_found", "type_mismatch", "timeout")
        error_message: Detailed error message
        screenshot_path: Optional path to screenshot captured at time of error
    """
    action: Action
    error_type: str
    error_message: str
    screenshot_path: Optional[str] = None


@dataclass
class ExecutionResult:
    """
    Result of executing an action batch.
    
    Provides feedback on batch execution success and details about any
    failures that occurred.
    
    Attributes:
        success: Whether the batch executed successfully
        actions_completed: Number of actions completed before stopping
        error: Error details if execution failed
        navigation_occurred: Whether any action in the batch caused navigation
    """
    success: bool
    actions_completed: int
    error: Optional[ExecutionError] = None
    navigation_occurred: bool = False


# ============================================================================
# Query Type Models
# ============================================================================

@dataclass
class ToolCall:
    """
    Direct action execution request via llm_call_step tool.
    
    Represents a request from GPT-4o to execute an action directly
    through the llm_call_step tool interface.
    
    Attributes:
        tool_name: Name of the tool to call (typically "llm_call_step")
        action_type: Type of action to execute (e.g., "click", "type", "select")
        parameters: Action-specific parameters
    """
    tool_name: str
    action_type: str
    parameters: Dict[str, Any]


@dataclass
class InfoSeekingQuery:
    """
    Straightforward element location request to the vision model.
    
    Sent by GPT-4o ONLY when there is a concrete plan to use the element.
    Requests the vision model to locate a specific element on the page.
    
    Attributes:
        element_description: Description of element to find (e.g., "find the create button")
        context: Why this information is needed (for logging and debugging)
    """
    element_description: str
    context: str


@dataclass
class ElementLocation:
    """
    Successful element location response from the vision model.
    
    Returned when the vision model successfully locates a requested element.
    
    Attributes:
        element_id: Unique identifier for the located element
        coordinates: (x, y) center coordinates of the element
        element_type: Type of the element (e.g., "button", "input")
        confidence: Confidence score for the location (0.0 to 1.0)
    """
    element_id: str
    coordinates: Tuple[int, int]
    element_type: str
    confidence: float


@dataclass
class ElementLocationFailure:
    """
    Failure response when vision model cannot locate a requested element.
    
    Reported back to GPT-4o to enable plan revision based on what
    elements are actually available.
    
    Attributes:
        element_description: What element was requested
        reason: Why the location failed (e.g., "not found", "ambiguous")
        available_elements: List of elements that ARE available on the page
    """
    element_description: str
    reason: str
    available_elements: List[str]


# ============================================================================
# Performance Tracking Models
# ============================================================================

@dataclass
class MetricsSummary:
    """
    Execution metrics summary for performance analysis.
    
    Tracks key performance indicators for task execution to measure
    the effectiveness of the adaptive agent system.
    
    Attributes:
        screenshot_count: Total number of screenshots captured
        vision_calls: Total number of vision model invocations
        orchestrator_calls: Total number of orchestrator model invocations
        plan_revisions: Total number of plan revisions
        total_actions: Total number of actions in all plans
        successful_actions: Number of actions that executed successfully
        execution_time_seconds: Total execution time in seconds
        actions_per_screenshot: Average actions executed per screenshot
    """
    screenshot_count: int
    vision_calls: int
    orchestrator_calls: int
    plan_revisions: int
    total_actions: int
    successful_actions: int
    execution_time_seconds: float
    actions_per_screenshot: float
