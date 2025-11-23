"""
Tests for backward compatibility with existing Task and TaskExecution infrastructure.

This module verifies that AdaptiveAgent properly integrates with:
- Task format (goal, url, evaluator)
- AgentBrowser interface
- ExperimentResult format
- TaskExecution workflow
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from arena.task import Task
from arena.agent import BaseAgent
from arena.state import AgentState
from arena.result import ExperimentResult
from agi_agents.adaptive_agent import AdaptiveAgent
from agi_agents.qwen.vision_model import QwenVisionModel
from agi_agents.orchestrator import GPT4Orchestrator
from agi_agents.metrics_tracker import MetricsTracker
from agi_agents.models import (
    ExecutionPlan,
    PageAnalysis,
    PageElement,
    Action,
)


@pytest.fixture
def mock_vision_model():
    """Create a mock vision model."""
    model = AsyncMock(spec=QwenVisionModel)
    model.analyze_page = AsyncMock(return_value=PageAnalysis(
        elements=[
            PageElement(
                element_id="btn1",
                element_type="button",
                label="Submit",
                coordinates=(100, 200),
                field_type=None,
                attributes={}
            )
        ],
        page_type="form",
        content_summary="A form with a submit button",
        timestamp=0.0
    ))
    return model


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    orchestrator = AsyncMock(spec=GPT4Orchestrator)
    orchestrator.analyze_and_plan = AsyncMock(return_value=ExecutionPlan(
        actions=[],
        reasoning="Task complete",
        created_at=0.0,
        revision_count=0
    ))
    return orchestrator


@pytest.fixture
def metrics_tracker():
    """Create a real metrics tracker."""
    return MetricsTracker()


@pytest.fixture
def adaptive_agent(mock_vision_model, mock_orchestrator, metrics_tracker):
    """Create an AdaptiveAgent instance."""
    return AdaptiveAgent(
        vision_model=mock_vision_model,
        orchestrator_model=mock_orchestrator,
        metrics_tracker=metrics_tracker
    )


@pytest.fixture
def mock_browser():
    """Create a mock browser."""
    browser = AsyncMock()
    browser.screenshot = AsyncMock(return_value="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    browser.page = MagicMock()
    browser.page.url = "https://example.com"
    return browser


class TestBackwardCompatibility:
    """Test backward compatibility with existing infrastructure."""
    
    def test_adaptive_agent_extends_base_agent(self, adaptive_agent):
        """
        Test that AdaptiveAgent properly extends BaseAgent.
        
        **Validates: Requirements 9.1**
        """
        assert isinstance(adaptive_agent, BaseAgent)
    
    @pytest.mark.asyncio
    async def test_step_method_is_async(self, adaptive_agent, mock_browser):
        """
        Test that step method is async for TaskExecution compatibility.
        
        **Validates: Requirements 9.4**
        """
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=0
        )
        
        # The step method should be awaitable
        result = adaptive_agent.step(mock_browser, state)
        assert hasattr(result, '__await__'), "step method should be async"
        
        # Actually await it
        result_state = await result
        assert isinstance(result_state, AgentState)
    
    @pytest.mark.asyncio
    async def test_agent_state_compatibility(self, adaptive_agent, mock_browser):
        """
        Test that AdaptiveAgent works with AgentState format.
        
        **Validates: Requirements 9.1**
        """
        state = AgentState(
            goal="Click the submit button",
            url="https://example.com",
            step=0,
            task_execution_id=123
        )
        
        result_state = await adaptive_agent.step(mock_browser, state)
        
        # Verify state attributes are preserved
        assert result_state.goal == "Click the submit button"
        assert result_state.url == "https://example.com"
        assert result_state.task_execution_id == 123
        assert isinstance(result_state.messages, list)
    
    @pytest.mark.asyncio
    async def test_finished_attribute_compatibility(self, adaptive_agent, mock_browser):
        """
        Test that AdaptiveAgent uses state.finished (not state.done).
        
        **Validates: Requirements 9.4**
        """
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=0
        )
        
        # Initially not finished
        assert state.finished is False
        
        result_state = await adaptive_agent.step(mock_browser, state)
        
        # Should have finished attribute (not done)
        assert hasattr(result_state, 'finished')
        assert not hasattr(result_state, 'done') or result_state.done is None
    
    @pytest.mark.asyncio
    async def test_browser_interface_compatibility(self, adaptive_agent, mock_browser):
        """
        Test that AdaptiveAgent uses existing browser automation primitives.
        
        **Validates: Requirements 9.2**
        """
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=0
        )
        
        await adaptive_agent.step(mock_browser, state)
        
        # Verify browser methods were called
        mock_browser.screenshot.assert_called()
    
    def test_metrics_summary_format(self, adaptive_agent):
        """
        Test that metrics summary has expected format.
        
        **Validates: Requirements 9.3**
        """
        # Start metrics tracking first
        adaptive_agent.metrics_tracker.start()
        
        summary = adaptive_agent.get_metrics_summary()
        
        # Verify summary has expected attributes
        assert hasattr(summary, 'screenshot_count')
        assert hasattr(summary, 'vision_calls')
        assert hasattr(summary, 'orchestrator_calls')
        assert hasattr(summary, 'plan_revisions')
    
    @pytest.mark.asyncio
    async def test_task_definition_parsing(self, adaptive_agent, mock_browser):
        """
        Test that agent works with Task format (goal and URL).
        
        **Validates: Requirements 9.1**
        """
        # Simulate how TaskExecution initializes state from Task
        task_goal = "Fill out the registration form"
        task_url = "https://example.com/register"
        
        state = AgentState(
            goal=task_goal,
            url=task_url,
            step=0
        )
        
        # Add initial message like TaskExecution does
        state.messages.append({
            "role": "user",
            "type": "user_input",
            "content": f"Your goal is {state.goal}"
        })
        
        result_state = await adaptive_agent.step(mock_browser, state)
        
        # Verify goal and URL are preserved
        assert result_state.goal == task_goal
        assert result_state.url == task_url
    
    @pytest.mark.asyncio
    async def test_error_handling_compatibility(self, adaptive_agent, mock_browser):
        """
        Test that agent handles errors in a way compatible with TaskExecution.
        
        **Validates: Requirements 9.4**
        """
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=0
        )
        
        # Even if internal operations fail, should return a valid state
        result_state = await adaptive_agent.step(mock_browser, state)
        
        # Should return AgentState (not raise exception)
        assert isinstance(result_state, AgentState)
        assert hasattr(result_state, 'error_count')
    
    @pytest.mark.asyncio
    async def test_messages_list_compatibility(self, adaptive_agent, mock_browser):
        """
        Test that agent properly maintains messages list in state.
        
        **Validates: Requirements 9.4**
        """
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=0,
            messages=[]
        )
        
        result_state = await adaptive_agent.step(mock_browser, state)
        
        # Messages should be a list
        assert isinstance(result_state.messages, list)
        # Should have added messages
        assert len(result_state.messages) > 0
        # Messages should have expected structure
        for msg in result_state.messages:
            assert isinstance(msg, dict)
            assert 'role' in msg or 'content' in msg
    
    @pytest.mark.asyncio
    async def test_step_counter_compatibility(self, adaptive_agent, mock_browser):
        """
        Test that agent properly handles step counter.
        
        **Validates: Requirements 9.4**
        """
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=5  # Start at step 5
        )
        
        result_state = await adaptive_agent.step(mock_browser, state)
        
        # Step should be incremented
        assert result_state.step > 5
