"""
Tests for AdaptiveAgent core workflow.

This module tests the basic functionality of the AdaptiveAgent class,
focusing on initialization and the parallel analysis workflow.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
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
)
from arena.state import AgentState
from arena.browser import AgentBrowser
from arena.image import Base64Image


class TestAdaptiveAgentInitialization:
    """Test AdaptiveAgent initialization."""
    
    def test_init_with_components(self):
        """Test that AdaptiveAgent initializes with required components."""
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Initialize agent
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        # Verify components are stored
        assert agent.vision_model is vision_model
        assert agent.orchestrator_model is orchestrator_model
        assert agent.metrics_tracker is metrics_tracker
        
        # Verify initial state
        assert agent.current_plan is not None
        assert len(agent.current_plan.actions) == 0
        assert agent.current_page_analysis is None


class TestAdaptiveAgentParallelAnalysis:
    """Test AdaptiveAgent parallel analysis workflow."""
    
    @pytest.mark.asyncio
    async def test_parallel_model_invocation(self):
        """Test that vision and orchestrator models are invoked in parallel."""
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
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
        
        # Create mock execution plan with NO actions to avoid execution
        mock_execution_plan = ExecutionPlan(
            actions=[],  # Empty plan to avoid action execution
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
        
        # Create a simple test image (1x1 red pixel)
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=0
        )
        
        # Execute step
        updated_state = await agent._async_step(browser, state)
        
        # Verify both models were called
        vision_model.analyze_page.assert_called_once()
        orchestrator_model.analyze_and_plan.assert_called_once()
        
        # Verify metrics were recorded
        assert metrics_tracker.record_screenshot.called
        assert metrics_tracker.record_vision_call.called
        assert metrics_tracker.record_orchestrator_call.called
        assert metrics_tracker.record_vision_call.called
        assert metrics_tracker.record_orchestrator_call.called
        
        # Verify state was updated
        assert updated_state.step == 1
        assert len(updated_state.messages) >= 2  # Should have page analysis and plan
        
        # Verify agent stored the results
        assert agent.current_page_analysis == mock_page_analysis
        assert agent.current_plan == mock_execution_plan
    
    @pytest.mark.asyncio
    async def test_parallel_analysis_result_combination(self):
        """Test that results from parallel analysis are properly combined."""
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Create mock page analysis with multiple elements
        mock_page_analysis = PageAnalysis(
            elements=[
                PageElement(
                    element_id="input_1",
                    element_type="input",
                    label="Username",
                    coordinates=(100, 100),
                    field_type="text"
                ),
                PageElement(
                    element_id="input_2",
                    element_type="input",
                    label="Password",
                    coordinates=(100, 150),
                    field_type="password"
                ),
                PageElement(
                    element_id="btn_1",
                    element_type="button",
                    label="Login",
                    coordinates=(100, 200),
                )
            ],
            page_type="form",
            content_summary="Login form with username and password",
            timestamp=1234567890.0
        )
        
        # Create mock execution plan with multiple actions
        mock_execution_plan = ExecutionPlan(
            actions=[
                Action(
                    action_type="type",
                    target_element_id="input_1",
                    parameters={"text": "testuser"},
                    is_navigation=False,
                    description="Enter username"
                ),
                Action(
                    action_type="type",
                    target_element_id="input_2",
                    parameters={"text": "testpass"},
                    is_navigation=False,
                    description="Enter password"
                ),
                Action(
                    action_type="click",
                    target_element_id="btn_1",
                    parameters={},
                    is_navigation=True,
                    description="Click login button"
                )
            ],
            reasoning="Fill login form and submit",
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
            goal="Login to the application",
            url="https://example.com/login",
            step=0
        )
        
        # Execute step
        updated_state = await agent._async_step(browser, state)
        
        # Verify both results are available in agent
        assert agent.current_page_analysis is not None
        assert agent.current_plan is not None
        
        # Verify page analysis has all elements
        assert len(agent.current_page_analysis.elements) == 3
        assert agent.current_page_analysis.page_type == "form"
        
        # Verify execution plan has all actions
        assert len(agent.current_plan.actions) == 3
        assert agent.current_plan.reasoning == "Fill login form and submit"
        
        # Verify state messages contain information from both models
        assert len(updated_state.messages) >= 2
        
        # Check that page analysis info is in messages
        page_analysis_msg = [m for m in updated_state.messages if "Page Analysis" in m.get("content", "")]
        assert len(page_analysis_msg) > 0
        assert page_analysis_msg[0]["page_type"] == "form"
        assert page_analysis_msg[0]["elements_count"] == 3
        
        # Check that execution plan info is in messages
        plan_msg = [m for m in updated_state.messages if "Execution Plan" in m.get("content", "")]
        assert len(plan_msg) > 0
        assert plan_msg[0]["actions_count"] == 3
        assert plan_msg[0]["revision_count"] == 0


class TestAdaptiveAgentMetricsTracking:
    """Test that AdaptiveAgent properly tracks metrics."""
    
    @pytest.mark.asyncio
    async def test_metrics_tracking_on_first_step(self):
        """Test that metrics tracking starts on first step."""
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Setup async mocks
        vision_model.analyze_page = AsyncMock(return_value=PageAnalysis(
            elements=[],
            page_type="unknown",
            content_summary="",
            timestamp=0.0
        ))
        orchestrator_model.analyze_and_plan = AsyncMock(return_value=ExecutionPlan(
            actions=[],
            reasoning="",
            created_at=0.0
        ))
        
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
            step=0  # First step
        )
        
        # Execute step
        await agent._async_step(browser, state)
        
        # Verify metrics tracking was started
        metrics_tracker.start.assert_called_once()
        
        # Verify metrics were recorded
        metrics_tracker.record_screenshot.assert_called_once()
        metrics_tracker.record_vision_call.assert_called_once()
        metrics_tracker.record_orchestrator_call.assert_called_once()


class TestAdaptiveAgentActionBatchExecution:
    """Test AdaptiveAgent action batch execution integration."""
    
    @pytest.mark.asyncio
    async def test_execute_action_batch_success_no_navigation(self):
        """Test successful action batch execution without navigation."""
        from agi_agents.models import ExecutionResult
        
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Create mock page analysis
        mock_page_analysis = PageAnalysis(
            elements=[
                PageElement(
                    element_id="input_1",
                    element_type="input",
                    label="Username",
                    coordinates=(100, 100),
                    field_type="text"
                ),
                PageElement(
                    element_id="input_2",
                    element_type="input",
                    label="Password",
                    coordinates=(100, 150),
                    field_type="password"
                )
            ],
            page_type="form",
            content_summary="Login form",
            timestamp=1234567890.0
        )
        
        # Create mock execution plan with non-navigation actions
        mock_execution_plan = ExecutionPlan(
            actions=[
                Action(
                    action_type="type",
                    target_element_id="input_1",
                    parameters={"text": "testuser"},
                    is_navigation=False,
                    description="Enter username"
                ),
                Action(
                    action_type="type",
                    target_element_id="input_2",
                    parameters={"text": "testpass"},
                    is_navigation=False,
                    description="Enter password"
                )
            ],
            reasoning="Fill login form",
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
        
        # Create mock browser
        browser = Mock(spec=AgentBrowser)
        browser.page = Mock()
        browser.page.mouse = Mock()
        browser.page.mouse.click = AsyncMock()
        browser.page.keyboard = Mock()
        browser.page.keyboard.type = AsyncMock()
        
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        state = AgentState(
            goal="Login to the application",
            url="https://example.com/login",
            step=0
        )
        
        # Execute step
        updated_state = await agent._async_step(browser, state)
        
        # Verify screenshot was captured once (no additional screenshot for non-navigation)
        assert browser.screenshot.call_count == 1
        
        # Verify actions were removed from plan
        assert len(agent.current_plan.actions) == 0
        
        # Verify success message in state
        success_msgs = [m for m in updated_state.messages if "Successfully executed" in m.get("content", "")]
        assert len(success_msgs) > 0
        assert success_msgs[0]["actions_completed"] == 2
        assert success_msgs[0]["navigation_occurred"] is False
    
    @pytest.mark.asyncio
    async def test_execute_action_batch_success_with_navigation(self):
        """Test successful action batch execution with navigation."""
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
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
            content_summary="Form with submit button",
            timestamp=1234567890.0
        )
        
        # Create mock execution plan with navigation action
        mock_execution_plan = ExecutionPlan(
            actions=[
                Action(
                    action_type="click",
                    target_element_id="btn_1",
                    parameters={},
                    is_navigation=True,
                    description="Click submit button"
                )
            ],
            reasoning="Submit the form",
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
        
        # Create mock browser
        browser = Mock(spec=AgentBrowser)
        browser.page = Mock()
        browser.page.mouse = Mock()
        browser.page.mouse.click = AsyncMock()
        
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        state = AgentState(
            goal="Submit the form",
            url="https://example.com/form",
            step=0
        )
        
        # Execute step
        updated_state = await agent._async_step(browser, state)
        
        # Verify screenshot was captured twice (initial + after navigation)
        assert browser.screenshot.call_count == 2
        
        # Verify metrics recorded two screenshots
        assert metrics_tracker.record_screenshot.call_count == 2
        
        # Verify success message indicates navigation
        success_msgs = [m for m in updated_state.messages if "Successfully executed" in m.get("content", "")]
        assert len(success_msgs) > 0
        assert success_msgs[0]["navigation_occurred"] is True
    
    @pytest.mark.asyncio
    async def test_execute_action_batch_error_captures_screenshot(self):
        """Test that batch execution errors trigger screenshot capture."""
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
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
            content_summary="Form",
            timestamp=1234567890.0
        )
        
        # Create mock execution plan
        mock_execution_plan = ExecutionPlan(
            actions=[
                Action(
                    action_type="click",
                    target_element_id="btn_missing",  # Element doesn't exist
                    parameters={},
                    is_navigation=False,
                    description="Click missing button"
                )
            ],
            reasoning="Try to click button",
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
        
        # Create mock browser
        browser = Mock(spec=AgentBrowser)
        browser.page = Mock()
        
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        state = AgentState(
            goal="Click button",
            url="https://example.com",
            step=0
        )
        
        # Execute step
        updated_state = await agent._async_step(browser, state)
        
        # Verify screenshot was captured twice (initial + after error)
        assert browser.screenshot.call_count == 2
        
        # Verify metrics recorded two screenshots
        assert metrics_tracker.record_screenshot.call_count == 2
        
        # Verify error message in state
        error_msgs = [m for m in updated_state.messages if "failed" in m.get("content", "").lower()]
        assert len(error_msgs) > 0
        assert "error_type" in error_msgs[0]
        assert "failed_action" in error_msgs[0]


class TestAdaptiveAgentQueryRouting:
    """Test AdaptiveAgent query routing and handling."""
    
    @pytest.mark.asyncio
    async def test_handle_tool_call(self):
        """Test that tool calls are properly handled via llm_call_step."""
        from agi_agents.models import ToolCall
        from PIL import Image
        import io
        
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Initialize agent
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        # Create mock browser and state
        browser = Mock(spec=AgentBrowser)
        browser.page = Mock()
        
        # Create a valid test image (1x1 red pixel JPEG)
        img = Image.new('RGB', (1, 1), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        test_image_bytes = base64.b64encode(img_bytes.read()).decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=1
        )
        
        # Create a tool call
        tool_call = ToolCall(
            tool_name="llm_call_step",
            action_type="click",
            parameters={"point_2d": [100, 200]}
        )
        
        # Mock the tool executor
        with patch('agi_agents.qwen.tools.QwenToolExecutor') as MockExecutor:
            mock_executor_instance = Mock()
            mock_executor_instance.execute_tool = AsyncMock(return_value="Clicked successfully")
            MockExecutor.return_value = mock_executor_instance
            
            # Execute tool call
            result = await agent._handle_tool_call(tool_call, browser, state)
            
            # Verify tool executor was created and called
            MockExecutor.assert_called_once()
            mock_executor_instance.execute_tool.assert_called_once_with(
                "click",
                {"point_2d": [100, 200]}
            )
            
            # Verify result
            assert result == "Clicked successfully"
    
    @pytest.mark.asyncio
    async def test_handle_info_seeking_query_success(self):
        """Test that info seeking queries route to vision model and return coordinates."""
        from agi_agents.models import InfoSeekingQuery, ElementLocation
        
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Initialize agent with page analysis
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        agent.current_page_analysis = PageAnalysis(
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
        
        # Create mock element location result
        mock_location = ElementLocation(
            element_id="btn_1",
            coordinates=(100, 200),
            element_type="button",
            confidence=0.95
        )
        
        vision_model.locate_element = AsyncMock(return_value=mock_location)
        
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=1
        )
        
        # Create an info seeking query
        query = InfoSeekingQuery(
            element_description="find the submit button",
            context="Need to submit the form"
        )
        
        # Execute query
        screenshot_bytes = b'\xff\x00\x00'
        result = await agent._handle_info_seeking_query(query, screenshot_bytes, state)
        
        # Verify vision model was called
        vision_model.locate_element.assert_called_once_with(
            screenshot=screenshot_bytes,
            element_description="find the submit button",
            page_analysis=agent.current_page_analysis
        )
        
        # Verify result contains coordinates
        assert "btn_1" in result
        assert "(100, 200)" in result
        assert "button" in result
    
    @pytest.mark.asyncio
    async def test_handle_info_seeking_query_failure(self):
        """Test that info seeking query failures are properly reported."""
        from agi_agents.models import InfoSeekingQuery, ElementLocationFailure
        
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Initialize agent with page analysis
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        agent.current_page_analysis = PageAnalysis(
            elements=[
                PageElement(
                    element_id="btn_1",
                    element_type="button",
                    label="Cancel",
                    coordinates=(100, 200),
                )
            ],
            page_type="form",
            content_summary="Test form",
            timestamp=1234567890.0
        )
        
        # Create mock element location failure
        mock_failure = ElementLocationFailure(
            element_description="find the submit button",
            reason="Element not found on page",
            available_elements=["Cancel"]
        )
        
        vision_model.locate_element = AsyncMock(return_value=mock_failure)
        
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=1
        )
        
        # Create an info seeking query
        query = InfoSeekingQuery(
            element_description="find the submit button",
            context="Need to submit the form"
        )
        
        # Execute query
        screenshot_bytes = b'\xff\x00\x00'
        result = await agent._handle_info_seeking_query(query, screenshot_bytes, state)
        
        # Verify vision model was called
        vision_model.locate_element.assert_called_once()
        
        # Verify result contains failure information
        assert "failed" in result.lower()
        assert "Element not found on page" in result
        assert "Cancel" in result
    
    @pytest.mark.asyncio
    async def test_route_query_tool_call(self):
        """Test that route_query correctly routes tool calls."""
        from agi_agents.models import ToolCall
        from PIL import Image
        import io
        
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Initialize agent
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        # Mock parse_output to return a ToolCall
        tool_call = ToolCall(
            tool_name="llm_call_step",
            action_type="click",
            parameters={"point_2d": [100, 200]}
        )
        orchestrator_model.parse_output = Mock(return_value=tool_call)
        
        # Create mock browser and state
        browser = Mock(spec=AgentBrowser)
        browser.page = Mock()
        
        # Create a valid test image (1x1 red pixel JPEG)
        img = Image.new('RGB', (1, 1), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        test_image_bytes = base64.b64encode(img_bytes.read()).decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=1
        )
        
        # Mock the tool executor
        with patch('agi_agents.qwen.tools.QwenToolExecutor') as MockExecutor:
            mock_executor_instance = Mock()
            mock_executor_instance.execute_tool = AsyncMock(return_value="Clicked successfully")
            MockExecutor.return_value = mock_executor_instance
            
            # Route the query - use valid JPEG bytes
            img_bytes.seek(0)
            screenshot_bytes = img_bytes.read()
            result, failure = await agent._route_query(
                "some response",
                browser,
                state,
                screenshot_bytes
            )
            
            # Verify parse_output was called
            orchestrator_model.parse_output.assert_called_once_with("some response")
            
            # Verify result is from tool call
            assert result == "Clicked successfully"
            assert failure is None
    
    @pytest.mark.asyncio
    async def test_route_query_info_seeking(self):
        """Test that route_query correctly routes info seeking queries."""
        from agi_agents.models import InfoSeekingQuery, ElementLocation
        
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Initialize agent with page analysis
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        
        agent.current_page_analysis = PageAnalysis(
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
        
        # Mock parse_output to return an InfoSeekingQuery
        query = InfoSeekingQuery(
            element_description="find the submit button",
            context="Need to submit the form"
        )
        orchestrator_model.parse_output = Mock(return_value=query)
        
        # Mock vision model response
        mock_location = ElementLocation(
            element_id="btn_1",
            coordinates=(100, 200),
            element_type="button",
            confidence=0.95
        )
        vision_model.locate_element = AsyncMock(return_value=mock_location)
        
        # Create mock browser and state
        browser = Mock(spec=AgentBrowser)
        state = AgentState(
            goal="Test goal",
            url="https://example.com",
            step=1
        )
        
        # Route the query
        screenshot_bytes = b'\xff\x00\x00'
        result, failure = await agent._route_query(
            "some response",
            browser,
            state,
            screenshot_bytes
        )
        
        # Verify parse_output was called
        orchestrator_model.parse_output.assert_called_once_with("some response")
        
        # Verify vision model was called
        assert vision_model.locate_element.call_count == 2  # Called twice in _route_query
        
        # Verify result contains location info
        assert "btn_1" in result
        assert failure is None



# ============================================================================
# Property-Based Tests
# ============================================================================

from hypothesis import given, strategies as st, settings


class TestAdaptiveAgentErrorHandlingProperties:
    """Property-based tests for error handling and plan revision."""
    
    @given(
        error_type=st.sampled_from([
            "element_not_found",
            "type_mismatch",
            "timeout",
            "navigation_error",
            "unknown_error"
        ]),
        error_message=st.text(min_size=1, max_size=200),
        actions_completed=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_24_error_details_passed_to_orchestrator(
        self,
        error_type,
        error_message,
        actions_completed
    ):
        """
        **Feature: hierarchical-action-batching, Property 24: Error details passed to Orchestrator**
        
        Property: For any execution error, the system should provide error details to the Orchestrator Model
        
        This test verifies that when an action execution fails, the error information
        (error type, error message, failed action details) is passed to the orchestrator
        for plan revision.
        
        **Validates: Requirements 6.2**
        """
        from agi_agents.models import ExecutionError, ExecutionResult
        
        # Create mock components
        vision_model = Mock(spec=QwenVisionModel)
        orchestrator_model = Mock(spec=GPT4Orchestrator)
        metrics_tracker = Mock(spec=MetricsTracker)
        
        # Create a mock page analysis
        mock_page_analysis = PageAnalysis(
            elements=[
                PageElement(
                    element_id="test_element",
                    element_type="button",
                    label="Test Button",
                    coordinates=(100, 200),
                )
            ],
            page_type="form",
            content_summary="Test page",
            timestamp=1234567890.0
        )
        
        # Create a failed action
        failed_action = Action(
            action_type="click",
            target_element_id="test_element",
            parameters={},
            is_navigation=False,
            description="Click test button"
        )
        
        # Create an execution error with the generated properties
        execution_error = ExecutionError(
            action=failed_action,
            error_type=error_type,
            error_message=error_message,
            screenshot_path=None
        )
        
        # Create mock execution plan with actions
        initial_plan = ExecutionPlan(
            actions=[failed_action],
            reasoning="Initial plan",
            created_at=1234567890.0,
            revision_count=0
        )
        
        # Create revised plan that orchestrator should return
        revised_plan = ExecutionPlan(
            actions=[],
            reasoning="Revised plan after error",
            created_at=1234567891.0,
            revision_count=1
        )
        
        # Setup async mocks
        vision_model.analyze_page = AsyncMock(return_value=mock_page_analysis)
        orchestrator_model.analyze_and_plan = AsyncMock(return_value=revised_plan)
        
        # Initialize agent with the initial plan
        agent = AdaptiveAgent(
            vision_model=vision_model,
            orchestrator_model=orchestrator_model,
            metrics_tracker=metrics_tracker
        )
        agent.current_plan = initial_plan
        agent.current_page_analysis = mock_page_analysis
        
        # Create mock browser that will fail during batch execution
        browser = Mock(spec=AgentBrowser)
        browser.page = Mock()
        
        test_image_bytes = base64.b64encode(b'\xff\x00\x00').decode('utf-8')
        browser.screenshot = AsyncMock(return_value=Base64Image(test_image_bytes))
        
        # Mock the batch executor to return a failure
        with patch('agi_agents.adaptive_agent.ActionBatchExecutor') as MockBatchExecutor:
            mock_batch_executor = Mock()
            mock_batch_executor.create_batch = Mock(return_value=[failed_action])
            mock_batch_executor.execute_batch = AsyncMock(return_value=ExecutionResult(
                success=False,
                actions_completed=actions_completed,
                error=execution_error,
                navigation_occurred=False
            ))
            MockBatchExecutor.return_value = mock_batch_executor
            
            state = AgentState(
                goal="Test goal",
                url="https://example.com",
                step=1
            )
            
            # Execute step - this should trigger error handling and plan revision
            updated_state = await agent._async_step(browser, state)
            
            # PROPERTY VERIFICATION:
            # The orchestrator's analyze_and_plan method should have been called
            # with the error_info parameter containing the execution error
            
            # Find the call that includes error_info
            calls_with_error = [
                call for call in orchestrator_model.analyze_and_plan.call_args_list
                if call.kwargs.get('error_info') is not None
            ]
            
            # Verify that at least one call included error information
            assert len(calls_with_error) > 0, (
                "Orchestrator should be called with error_info when execution fails"
            )
            
            # Get the error_info that was passed
            error_info_passed = calls_with_error[0].kwargs['error_info']
            
            # Verify all error details were passed correctly
            assert error_info_passed is not None, "error_info should not be None"
            assert error_info_passed.error_type == error_type, (
                f"Error type should be '{error_type}', got '{error_info_passed.error_type}'"
            )
            assert error_info_passed.error_message == error_message, (
                f"Error message should be '{error_message}', got '{error_info_passed.error_message}'"
            )
            assert error_info_passed.action == failed_action, (
                "Failed action should be included in error info"
            )
            
            # Verify plan revision was recorded
            assert metrics_tracker.record_plan_revision.called, (
                "Plan revision should be recorded in metrics"
            )
            
            # Verify the plan was updated
            assert agent.current_plan == revised_plan, (
                "Agent's current plan should be updated to the revised plan"
            )
            
            # Verify consecutive revisions counter was incremented
            assert agent.consecutive_revisions == 1, (
                "Consecutive revisions counter should be incremented"
            )
