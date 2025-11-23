"""
Tests for ActionBatchExecutor.

These tests verify the core functionality of action batching:
- Extracting consecutive same-page actions
- Stopping at navigation actions
- Sequential execution
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from agi_agents.batch_executor import ActionBatchExecutor
from agi_agents.models import (
    Action,
    ExecutionPlan,
    PageAnalysis,
    PageElement,
)


@pytest.fixture
def mock_browser():
    """Create a mock browser with page and mouse/keyboard."""
    browser = Mock()
    page = AsyncMock()
    
    # Mock mouse and keyboard
    page.mouse = AsyncMock()
    page.mouse.click = AsyncMock()
    page.mouse.move = AsyncMock()
    page.mouse.wheel = AsyncMock()
    
    page.keyboard = AsyncMock()
    page.keyboard.type = AsyncMock()
    
    page.wait_for_timeout = AsyncMock()
    page.goto = AsyncMock()
    page.evaluate = AsyncMock(return_value={"width": 1280, "height": 720})
    page.query_selector_all = AsyncMock(return_value=[])
    
    browser.page = page
    return browser


@pytest.fixture
def sample_page_analysis():
    """Create a sample page analysis with elements."""
    return PageAnalysis(
        elements=[
            PageElement(
                element_id="btn_1",
                element_type="button",
                label="Submit",
                coordinates=(100, 200),
            ),
            PageElement(
                element_id="input_1",
                element_type="input",
                label="Username",
                coordinates=(150, 100),
                field_type="text",
            ),
            PageElement(
                element_id="select_1",
                element_type="select",
                label="Country",
                coordinates=(200, 150),
            ),
        ],
        page_type="form",
        content_summary="Login form",
        timestamp=1234567890.0,
    )


def test_create_batch_empty_plan(mock_browser):
    """Test creating batch from empty plan."""
    executor = ActionBatchExecutor(mock_browser)
    plan = ExecutionPlan(actions=[])
    
    batch = executor.create_batch(plan)
    
    assert batch == []


def test_create_batch_single_action(mock_browser):
    """Test creating batch with single action."""
    executor = ActionBatchExecutor(mock_browser)
    action = Action(
        action_type="click",
        target_element_id="btn_1",
        parameters={},
        is_navigation=False,
        description="Click button",
    )
    plan = ExecutionPlan(actions=[action])
    
    batch = executor.create_batch(plan)
    
    assert len(batch) == 1
    assert batch[0] == action


def test_create_batch_stops_at_navigation(mock_browser):
    """Test that batch creation stops at navigation action."""
    executor = ActionBatchExecutor(mock_browser)
    
    actions = [
        Action(
            action_type="type",
            target_element_id="input_1",
            parameters={"text": "user"},
            is_navigation=False,
            description="Type username",
        ),
        Action(
            action_type="click",
            target_element_id="btn_1",
            parameters={},
            is_navigation=True,  # Navigation action
            description="Click submit",
        ),
        Action(
            action_type="type",
            target_element_id="input_2",
            parameters={"text": "more"},
            is_navigation=False,
            description="Type more",
        ),
    ]
    plan = ExecutionPlan(actions=actions)
    
    batch = executor.create_batch(plan)
    
    # Should include first two actions (up to and including navigation)
    assert len(batch) == 2
    assert batch[0].action_type == "type"
    assert batch[1].action_type == "click"
    assert batch[1].is_navigation is True


def test_create_batch_all_same_page(mock_browser):
    """Test batch with all same-page actions."""
    executor = ActionBatchExecutor(mock_browser)
    
    actions = [
        Action(
            action_type="type",
            target_element_id="input_1",
            parameters={"text": "user"},
            is_navigation=False,
            description="Type username",
        ),
        Action(
            action_type="type",
            target_element_id="input_2",
            parameters={"text": "pass"},
            is_navigation=False,
            description="Type password",
        ),
        Action(
            action_type="select",
            target_element_id="select_1",
            parameters={"value": "US"},
            is_navigation=False,
            description="Select country",
        ),
    ]
    plan = ExecutionPlan(actions=actions)
    
    batch = executor.create_batch(plan)
    
    # Should include all actions since none cause navigation
    assert len(batch) == 3


@pytest.mark.asyncio
async def test_execute_batch_empty(mock_browser, sample_page_analysis):
    """Test executing empty batch."""
    executor = ActionBatchExecutor(mock_browser)
    
    result = await executor.execute_batch([], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 0
    assert result.navigation_occurred is False
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_batch_single_click(mock_browser, sample_page_analysis):
    """Test executing single click action."""
    executor = ActionBatchExecutor(mock_browser)
    
    action = Action(
        action_type="click",
        target_element_id="btn_1",
        parameters={},
        is_navigation=False,
        description="Click button",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
    assert result.navigation_occurred is False
    assert result.error is None
    
    # Verify click was called with correct coordinates
    mock_browser.page.mouse.click.assert_called_once_with(100, 200)


@pytest.mark.asyncio
async def test_execute_batch_type_action(mock_browser, sample_page_analysis):
    """Test executing type action."""
    executor = ActionBatchExecutor(mock_browser)
    
    action = Action(
        action_type="type",
        target_element_id="input_1",
        parameters={"text": "testuser"},
        is_navigation=False,
        description="Type username",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
    
    # Verify click (to focus) and type were called
    mock_browser.page.mouse.click.assert_called_once_with(150, 100)
    mock_browser.page.keyboard.type.assert_called_once_with("testuser", delay=50)


@pytest.mark.asyncio
async def test_execute_batch_multiple_actions(mock_browser, sample_page_analysis):
    """Test executing multiple actions in sequence."""
    executor = ActionBatchExecutor(mock_browser)
    
    actions = [
        Action(
            action_type="type",
            target_element_id="input_1",
            parameters={"text": "user"},
            is_navigation=False,
            description="Type username",
        ),
        Action(
            action_type="click",
            target_element_id="btn_1",
            parameters={},
            is_navigation=False,
            description="Click button",
        ),
    ]
    
    result = await executor.execute_batch(actions, sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 2
    assert result.navigation_occurred is False
    
    # Verify both actions were executed
    assert mock_browser.page.keyboard.type.call_count == 1
    assert mock_browser.page.mouse.click.call_count == 2  # Once for focus, once for click


@pytest.mark.asyncio
async def test_execute_batch_tracks_navigation(mock_browser, sample_page_analysis):
    """Test that navigation is tracked correctly."""
    executor = ActionBatchExecutor(mock_browser)
    
    action = Action(
        action_type="click",
        target_element_id="btn_1",
        parameters={},
        is_navigation=True,  # This action causes navigation
        description="Click submit",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
    assert result.navigation_occurred is True


@pytest.mark.asyncio
async def test_execute_batch_stops_on_error(mock_browser, sample_page_analysis):
    """Test that batch execution stops on first error."""
    executor = ActionBatchExecutor(mock_browser)
    
    # Make the second action fail
    mock_browser.page.mouse.click.side_effect = [
        None,  # First click succeeds
        Exception("Element not clickable"),  # Second click fails
    ]
    
    actions = [
        Action(
            action_type="click",
            target_element_id="btn_1",
            parameters={},
            is_navigation=False,
            description="Click button 1",
        ),
        Action(
            action_type="click",
            target_element_id="btn_1",
            parameters={},
            is_navigation=False,
            description="Click button 2",
        ),
        Action(
            action_type="click",
            target_element_id="btn_1",
            parameters={},
            is_navigation=False,
            description="Click button 3",
        ),
    ]
    
    result = await executor.execute_batch(actions, sample_page_analysis)
    
    assert result.success is False
    assert result.actions_completed == 1  # Only first action completed
    assert result.error is not None
    assert result.error.error_type == "unknown_error"
    assert "not clickable" in result.error.error_message


@pytest.mark.asyncio
async def test_execute_batch_element_not_found(mock_browser, sample_page_analysis):
    """Test error handling when element is not found."""
    executor = ActionBatchExecutor(mock_browser)
    
    action = Action(
        action_type="click",
        target_element_id="nonexistent",
        parameters={},
        is_navigation=False,
        description="Click nonexistent",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is False
    assert result.actions_completed == 0
    assert result.error is not None
    assert result.error.error_type == "element_not_found"


@pytest.mark.asyncio
async def test_execute_scroll_action(mock_browser, sample_page_analysis):
    """Test executing scroll action."""
    executor = ActionBatchExecutor(mock_browser)
    
    action = Action(
        action_type="scroll",
        target_element_id="",
        parameters={"direction": "down", "pixels": 500},
        is_navigation=False,
        description="Scroll down",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
    
    # Verify scroll was called
    mock_browser.page.mouse.wheel.assert_called_once_with(0, 500)


@pytest.mark.asyncio
async def test_execute_wait_action(mock_browser, sample_page_analysis):
    """Test executing wait action."""
    executor = ActionBatchExecutor(mock_browser)
    
    action = Action(
        action_type="wait",
        target_element_id="",
        parameters={"duration": 2000},
        is_navigation=False,
        description="Wait 2 seconds",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
    
    # Verify wait was called
    mock_browser.page.wait_for_timeout.assert_called_once_with(2000)


@pytest.mark.asyncio
async def test_execute_goto_action(mock_browser, sample_page_analysis):
    """Test executing goto action."""
    executor = ActionBatchExecutor(mock_browser)
    
    action = Action(
        action_type="goto",
        target_element_id="",
        parameters={"url": "https://example.com"},
        is_navigation=True,
        description="Navigate to example.com",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
    assert result.navigation_occurred is True
    
    # Verify goto was called
    mock_browser.page.goto.assert_called_once_with("https://example.com")


# ============================================================================
# Action Selection Tests
# ============================================================================

def test_select_action_type_text_input(mock_browser):
    """Test action selection for text input elements."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="input_1",
        element_type="input",
        label="Username",
        coordinates=(100, 100),
        field_type="text",
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "type"


def test_select_action_type_email_input(mock_browser):
    """Test action selection for email input elements."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="input_2",
        element_type="input",
        label="Email",
        coordinates=(100, 100),
        field_type="email",
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "type"


def test_select_action_type_password_input(mock_browser):
    """Test action selection for password input elements."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="input_3",
        element_type="input",
        label="Password",
        coordinates=(100, 100),
        field_type="password",
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "type"


def test_select_action_type_dropdown(mock_browser):
    """Test action selection for dropdown/select elements."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="select_1",
        element_type="select",
        label="Country",
        coordinates=(100, 100),
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "select"


def test_select_action_type_date_picker(mock_browser):
    """Test action selection for date picker elements."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="date_1",
        element_type="input",
        label="Birth Date",
        coordinates=(100, 100),
        field_type="date",
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "date_input"


def test_select_action_type_datetime_picker(mock_browser):
    """Test action selection for datetime picker elements."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="datetime_1",
        element_type="input",
        label="Appointment Time",
        coordinates=(100, 100),
        field_type="datetime-local",
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "date_input"


def test_select_action_type_button(mock_browser):
    """Test action selection for button elements (default to click)."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="btn_1",
        element_type="button",
        label="Submit",
        coordinates=(100, 100),
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "click"


def test_select_action_type_link(mock_browser):
    """Test action selection for link elements (default to click)."""
    executor = ActionBatchExecutor(mock_browser)
    
    element = PageElement(
        element_id="link_1",
        element_type="link",
        label="Learn More",
        coordinates=(100, 100),
    )
    
    action_type = executor.select_action_type(element)
    assert action_type == "click"


@pytest.mark.asyncio
async def test_action_type_validation_mismatch(mock_browser, sample_page_analysis):
    """Test that action type mismatch raises an error."""
    executor = ActionBatchExecutor(mock_browser)
    
    # Try to use 'type' action on a select element
    action = Action(
        action_type="type",
        target_element_id="select_1",
        parameters={"text": "some text"},
        is_navigation=False,
        description="Type into dropdown (should fail)",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is False
    assert result.actions_completed == 0
    assert result.error is not None
    assert "type mismatch" in result.error.error_type.lower() or "mismatch" in result.error.error_message.lower()


@pytest.mark.asyncio
async def test_date_input_action(mock_browser):
    """Test executing date input action."""
    executor = ActionBatchExecutor(mock_browser)
    
    page_analysis = PageAnalysis(
        elements=[
            PageElement(
                element_id="date_1",
                element_type="input",
                label="Birth Date",
                coordinates=(100, 100),
                field_type="date",
            ),
        ],
        page_type="form",
        content_summary="Date form",
        timestamp=1234567890.0,
    )
    
    action = Action(
        action_type="date_input",
        target_element_id="date_1",
        parameters={"value": "2024-01-15"},
        is_navigation=False,
        description="Enter birth date",
    )
    
    result = await executor.execute_batch([action], page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
    
    # Verify date input was called
    mock_browser.page.mouse.click.assert_called_once_with(100, 100)
    mock_browser.page.keyboard.type.assert_called_once()


@pytest.mark.asyncio
async def test_action_type_validation_correct_type(mock_browser, sample_page_analysis):
    """Test that correct action type passes validation."""
    executor = ActionBatchExecutor(mock_browser)
    
    # Use 'type' action on a text input element (correct)
    action = Action(
        action_type="type",
        target_element_id="input_1",
        parameters={"text": "username"},
        is_navigation=False,
        description="Type username",
    )
    
    result = await executor.execute_batch([action], sample_page_analysis)
    
    assert result.success is True
    assert result.actions_completed == 1
