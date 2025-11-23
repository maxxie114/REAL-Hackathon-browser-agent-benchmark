"""
Tests for performance optimizations in AdaptiveAgent.

This module tests the performance optimization features implemented in task 10:
- Skipping orchestrator calls when plan has valid actions
- Ensuring vision model is called once per screenshot
- Metrics tracking throughout the workflow
"""

import pytest
from unittest.mock import Mock, AsyncMock
import base64

from agi_agents.adaptive_agent import AdaptiveAgent
from agi_agents.qwen.vision_model import QwenVisionModel
from agi_agents.orchestrator import GPT4Orchestrator
from agi_agents.metrics_tracker import MetricsTracker
from agi_agents.models import (
    PageAnalysis,
    PageElement,
    ExecutionPlan,
    Action,
    ExecutionResult,
)
from arena.state import AgentState
from arena.browser import AgentBrowser
from arena.image import Base64Image


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_skipped_when_plan_has_actions(self):
        """
        Test that orchestrator is not called when plan has valid actions.
        Validates Requirement 7.2: Skip orchestrator when plan has valid actions.
        """
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = MetricsTracker()
        
        # Create mock page analysis
        mock_page_analysis = PageAnalysis(
            elements=[
                PageElement(
                    element_id="btn_1",
                    element_type="button",
                    label="Submit",
                    coordinates=(100, 200),
                )
            ],
            page_type="form",
            content_summary="Test form",
            timestamp=1234567890.0
        )
        
        # Setup async mocks
        vision_model.analyze_page = AsyncMock(return_value=mock_page_analysis)
        
        # Initialize agent with a pre-existing plan
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        # Set up a plan with actions (simulating previous step)
        agent.current_plan = ExecutionPlan(
            actions=[
                Action(
                    action_type="click",
                    target_element_id="btn_1",
                    parameters={},
                    is_navigation=False,
                    description="Click submit button"
                )
            ],
            reasoning="Test plan",
            created_at=1234567890.0,
            revision_count=0
        )
        agent.current_page_analysis = mock_page_analysis
        
        # Create mock browser and state
        browser = Mock(spec=AgentBrowser)
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        # Mock the batch executor to prevent actual execution
        from unittest.mock import patch
        from agi_agents.batch_executor import ActionBatchExecutor
        with patch.object(ActionBatchExecutor, 'execute_batch') as mock_execute:
            mock_execute.return_value = ExecutionResult(
                success=True,
                actions_completed=1,
                error=None,
                navigation_occurred=False
            )
            
            state = AgentState(
                goal="Test goal",
                url="https://example.com",
                step=1  # Not the first step
            )
            
            # Execute step
            await agent._async_step(browser, state)
        
        # Verify orchestrator was NOT called (optimization working)
        orchestrator_model.analyze_and_plan.assert_not_called()
        
        # Verify vision model WAS called (once per screenshot)
        vision_model.analyze_page.assert_called_once()
        
        # Verify metrics show only 1 vision call and 0 orchestrator calls
        assert metrics_tracker.vision_calls == 1
        assert metrics_tracker.orchestrator_calls == 0
    
    @pytest.mark.asyncio
    async def test_vision_model_called_once_per_screenshot(self):
        """
        Test that vision model is called exactly once per screenshot.
        Validates Requirement 7.1: Vision model called once per screenshot.
        """
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = MetricsTracker()
        
        # Create mock page analysis
        mock_page_analysis = PageAnalysis(
            elements=[],
            page_type="form",
            content_summary="Test form",
            timestamp=1234567890.0
        )
        
        # Create mock execution plan
        mock_execution_plan = ExecutionPlan(
            actions=[],
            reasoning="Test plan",
            created_at=1234567890.0,
            revision_count=0
        )
        
        # Setup async mocks
        vision_model.analyze_page = AsyncMock(return_value=mock_page_analysis)
        orchestrator_model.analyze_and_plan = AsyncMock(return_value=mock_execution_plan)
        
        # Initialize agent
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        # Create mock browser and state
        browser = Mock(spec=AgentBrowser)
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=0
        )
        
        # Execute step
        await agent._async_step(browser, state)
        
        # Verify vision model was called exactly once
        assert vision_model.analyze_page.call_count == 1
        
        # Verify screenshot count matches vision call count
        assert metrics_tracker.screenshot_count == 1
        assert metrics_tracker.vision_calls == 1
        assert metrics_tracker.screenshot_count == metrics_tracker.vision_calls
    
    @pytest.mark.asyncio
    async def test_metrics_recorded_throughout_workflow(self):
        """
        Test that metrics are recorded at all appropriate points.
        Validates Requirements 8.1-8.5: Track all metrics.
        """
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = MetricsTracker()
        
        # Create mock page analysis
        mock_page_analysis = PageAnalysis(
            elements=[
                PageElement(
                    element_id="btn_1",
                    element_type="button",
                    label="Submit",
                    coordinates=(100, 200),
                )
            ],
            page_type="form",
            content_summary="Test form",
            timestamp=1234567890.0
        )
        
        # Create mock execution plan
        mock_execution_plan = ExecutionPlan(
            actions=[
                Action(
                    action_type="click",
                    target_element_id="btn_1",
                    parameters={},
                    is_navigation=False,
                    description="Click submit button"
                )
            ],
            reasoning="Test plan",
            created_at=1234567890.0,
            revision_count=0
        )
        
        # Setup async mocks
        vision_model.analyze_page = AsyncMock(return_value=mock_page_analysis)
        orchestrator_model.analyze_and_plan = AsyncMock(return_value=mock_execution_plan)
        
        # Initialize agent
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        # Create mock browser and state
        browser = Mock(spec=AgentBrowser)
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        # Mock the batch executor
        from unittest.mock import patch
        from agi_agents.batch_executor import ActionBatchExecutor
        with patch.object(ActionBatchExecutor, 'execute_batch') as mock_execute:
            mock_execute.return_value = ExecutionResult(
                success=True,
                actions_completed=1,
                error=None,
                navigation_occurred=False
            )
            
            state = AgentState(
                goal="Test goal",
                url="https://example.com",
                step=0
            )
            
            # Execute step
            await agent._async_step(browser, state)
        
        # Verify all metrics were recorded
        assert metrics_tracker.screenshot_count == 1  # Requirement 8.1
        assert metrics_tracker.vision_calls == 1  # Requirement 8.2
        assert metrics_tracker.orchestrator_calls == 1  # Requirement 8.3
        assert metrics_tracker.total_actions == 1  # Action tracking
        assert metrics_tracker.successful_actions == 1  # Action tracking
        
        # Verify we can get a summary
        summary = metrics_tracker.get_summary()
        assert summary.screenshot_count == 1
        assert summary.vision_calls == 1
        assert summary.orchestrator_calls == 1
        assert summary.total_actions == 1
        assert summary.successful_actions == 1
    
    @pytest.mark.asyncio
    async def test_get_metrics_summary_method(self):
        """Test that agent provides get_metrics_summary method."""
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = MetricsTracker()
        
        # Initialize agent
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        # Start tracking
        metrics_tracker.start()
        
        # Record some metrics
        metrics_tracker.record_screenshot()
        metrics_tracker.record_vision_call()
        metrics_tracker.record_orchestrator_call()
        metrics_tracker.record_actions(total=5, successful=4)
        
        # Get summary via agent method
        summary = agent.get_metrics_summary()
        
        # Verify summary contains expected data
        assert summary.screenshot_count == 1
        assert summary.vision_calls == 1
        assert summary.orchestrator_calls == 1
        assert summary.total_actions == 5
        assert summary.successful_actions == 4
