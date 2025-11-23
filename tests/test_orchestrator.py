"""
Basic tests for GPT4Orchestrator implementation.

These tests verify the core functionality of the orchestrator including
plan generation, error handling, and query parsing.
"""

import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from agi_agents.orchestrator import GPT4Orchestrator
from agi_agents.models import (
    Action,
    ExecutionPlan,
    ExecutionError,
    PageAnalysis,
    PageElement,
    ElementLocationFailure,
    ToolCall,
    InfoSeekingQuery,
)


@pytest.fixture
def orchestrator():
    """Create orchestrator with mocked client."""
    mock_client = AsyncMock()
    return GPT4Orchestrator(client=mock_client)


@pytest.fixture
def sample_page_analysis():
    """Create sample page analysis."""
    return PageAnalysis(
        elements=[
            PageElement(
                element_id="btn_1",
                element_type="button",
                label="Submit",
                coordinates=(450, 300),
            ),
            PageElement(
                element_id="input_1",
                element_type="input",
                label="Username",
                coordinates=(450, 200),
                field_type="text",
            ),
        ],
        page_type="form",
        content_summary="Login form",
        timestamp=time.time(),
    )


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator can be initialized."""
    orchestrator = GPT4Orchestrator(
        model="openai/gpt-4o",
        api_key="test-key",
        base_url="https://test.com",
    )
    assert orchestrator.model == "openai/gpt-4o"
    assert orchestrator.client is not None


@pytest.mark.asyncio
async def test_analyze_and_plan_creates_plan(orchestrator, sample_page_analysis):
    """Test that analyze_and_plan creates an execution plan."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""```json
{
  "plan": {
    "actions": [
      {
        "action_type": "type",
        "target_element_id": "input_1",
        "parameters": {"text": "testuser"},
        "is_navigation": false,
        "description": "Enter username"
      },
      {
        "action_type": "click",
        "target_element_id": "btn_1",
        "parameters": {},
        "is_navigation": true,
        "description": "Click submit button"
      }
    ],
    "reasoning": "Fill in username and submit form"
  }
}
```"""
            )
        )
    ]
    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create plan
    empty_plan = ExecutionPlan(actions=[])
    result = await orchestrator.analyze_and_plan(
        goal="Log in with username 'testuser'",
        current_plan=empty_plan,
        page_analysis=sample_page_analysis,
        url="https://real-gomail.vercel.app/"
    )
    
    # Verify plan was created
    assert isinstance(result, ExecutionPlan)
    assert len(result.actions) == 2
    assert result.actions[0].action_type == "type"
    assert result.actions[0].target_element_id == "input_1"
    assert result.actions[1].action_type == "click"
    assert result.actions[1].target_element_id == "btn_1"
    assert result.reasoning == "Fill in username and submit form"


@pytest.mark.asyncio
async def test_analyze_and_plan_increments_revision_on_error(
    orchestrator, sample_page_analysis
):
    """Test that revision count increments when error occurs."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""```json
{
  "plan": {
    "actions": [
      {
        "action_type": "click",
        "target_element_id": "btn_1",
        "parameters": {},
        "is_navigation": true,
        "description": "Click submit button"
      }
    ],
    "reasoning": "Revised plan after error"
  }
}
```"""
            )
        )
    ]
    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create plan with existing revision count
    current_plan = ExecutionPlan(actions=[], revision_count=1)
    
    # Create error info
    error_info = ExecutionError(
        action=Action(
            action_type="type",
            target_element_id="input_1",
            parameters={"text": "test"},
            is_navigation=False,
            description="Type into input",
        ),
        error_type="element_not_found",
        error_message="Element not found",
    )
    
    # Revise plan
    result = await orchestrator.analyze_and_plan(
        goal="Log in",
        current_plan=current_plan,
        page_analysis=sample_page_analysis,
        error_info=error_info,
    )
    
    # Verify revision count incremented
    assert result.revision_count == 2


@pytest.mark.asyncio
async def test_analyze_and_plan_handles_location_failure(
    orchestrator, sample_page_analysis
):
    """Test that plan revision handles element location failures."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""```json
{
  "plan": {
    "actions": [
      {
        "action_type": "click",
        "target_element_id": "btn_1",
        "parameters": {},
        "is_navigation": true,
        "description": "Click available submit button"
      }
    ],
    "reasoning": "Using available button instead"
  }
}
```"""
            )
        )
    ]
    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create location failure
    location_failure = ElementLocationFailure(
        element_description="find the login button",
        reason="Element not found",
        available_elements=["Submit", "Username"],
    )
    
    current_plan = ExecutionPlan(actions=[])
    
    # Revise plan
    result = await orchestrator.analyze_and_plan(
        goal="Log in",
        current_plan=current_plan,
        page_analysis=sample_page_analysis,
        location_failure=location_failure,
    )
    
    # Verify plan was revised
    assert isinstance(result, ExecutionPlan)
    assert len(result.actions) == 1
    assert result.revision_count == 1


def test_parse_output_tool_call():
    """Test parsing tool call from orchestrator output."""
    orchestrator = GPT4Orchestrator(api_key="test")
    
    response = """```json
{
  "tool_call": {
    "tool_name": "llm_call_step",
    "action_type": "click",
    "parameters": {"element_id": "btn_1"}
  }
}
```"""
    
    result = orchestrator.parse_output(response)
    
    assert isinstance(result, ToolCall)
    assert result.tool_name == "llm_call_step"
    assert result.action_type == "click"
    assert result.parameters == {"element_id": "btn_1"}


def test_parse_output_info_seeking_query():
    """Test parsing info seeking query from orchestrator output."""
    orchestrator = GPT4Orchestrator(api_key="test")
    
    response = """```json
{
  "info_query": {
    "element_description": "find the submit button",
    "context": "Need to locate submit button to complete form"
  }
}
```"""
    
    result = orchestrator.parse_output(response)
    
    assert isinstance(result, InfoSeekingQuery)
    assert result.element_description == "find the submit button"
    assert result.context == "Need to locate submit button to complete form"


def test_parse_output_invalid_response():
    """Test that invalid response raises ValueError."""
    orchestrator = GPT4Orchestrator(api_key="test")
    
    response = """```json
{
  "invalid": "response"
}
```"""
    
    with pytest.raises(ValueError, match="must contain either"):
        orchestrator.parse_output(response)


@pytest.mark.asyncio
async def test_analyze_and_plan_retries_on_failure(orchestrator, sample_page_analysis):
    """Test that analyze_and_plan retries on API failure."""
    # Mock API to fail twice then succeed
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""```json
{
  "plan": {
    "actions": [],
    "reasoning": "Success after retries"
  }
}
```"""
            )
        )
    ]
    
    orchestrator.client.chat.completions.create = AsyncMock(
        side_effect=[
            Exception("API error 1"),
            Exception("API error 2"),
            mock_response,
        ]
    )
    
    # Should succeed after retries
    result = await orchestrator.analyze_and_plan(
        goal="Test goal",
        current_plan=ExecutionPlan(actions=[]),
        page_analysis=sample_page_analysis,
    )
    
    assert isinstance(result, ExecutionPlan)
    assert result.reasoning == "Success after retries"
    
    # Verify it tried 3 times
    assert orchestrator.client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_analyze_and_plan_fails_after_max_retries(
    orchestrator, sample_page_analysis
):
    """Test that analyze_and_plan fails after max retries."""
    # Mock API to always fail
    orchestrator.client.chat.completions.create = AsyncMock(
        side_effect=Exception("API error")
    )
    
    # Should raise RuntimeError after max retries
    with pytest.raises(RuntimeError, match="API call failed"):
        await orchestrator.analyze_and_plan(
            goal="Test goal",
            current_plan=ExecutionPlan(actions=[]),
            page_analysis=sample_page_analysis,
        )


@pytest.mark.asyncio
async def test_turn_1_plan_limits_to_single_action(orchestrator, sample_page_analysis):
    """Test that turn 1 plans are limited to at most one exploratory action."""
    # Mock API response with multiple actions
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "plan": {
            "actions": [
                {
                    "action_type": "goto",
                    "target_element_id": "page",
                    "parameters": {"url": "https://example.com"},
                    "is_navigation": True,
                    "description": "Navigate to example.com"
                },
                {
                    "action_type": "wait",
                    "target_element_id": "page",
                    "parameters": {"seconds": 2},
                    "is_navigation": False,
                    "description": "Wait for page load"
                }
            ],
            "reasoning": "Navigate and wait"
        }
    })
    
    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create plan with turn 1 flag
    empty_plan = ExecutionPlan(actions=[])
    result = await orchestrator.analyze_and_plan(
        goal="Visit example.com",
        current_plan=empty_plan,
        page_analysis=sample_page_analysis,
        is_turn_1=True
    )
    
    # Verify plan was limited to 1 action
    assert len(result.actions) == 1
    assert result.actions[0].action_type == "goto"


@pytest.mark.asyncio
async def test_turn_1_plan_removes_invalid_actions(orchestrator, sample_page_analysis):
    """Test that turn 1 plans remove invalid actions like click, type, select."""
    # Mock API response with invalid turn 1 actions
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "plan": {
            "actions": [
                {
                    "action_type": "click",
                    "target_element_id": "login_button",
                    "parameters": {},
                    "is_navigation": False,
                    "description": "Click login button"
                }
            ],
            "reasoning": "Click the login button"
        }
    })
    
    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create plan with turn 1 flag
    empty_plan = ExecutionPlan(actions=[])
    result = await orchestrator.analyze_and_plan(
        goal="Log in",
        current_plan=empty_plan,
        page_analysis=sample_page_analysis,
        is_turn_1=True
    )
    
    # Verify invalid action was removed and replaced with default wait
    assert len(result.actions) == 1
    assert result.actions[0].action_type == "wait"
    assert "turn 1" in result.reasoning.lower() or "Turn 1" in result.reasoning


@pytest.mark.asyncio
async def test_turn_1_plan_allows_valid_exploratory_actions(orchestrator, sample_page_analysis):
    """Test that turn 1 plans allow valid exploratory actions (goto, wait, scroll)."""
    # Mock API response with valid turn 1 action
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "plan": {
            "actions": [
                {
                    "action_type": "scroll",
                    "target_element_id": "page",
                    "parameters": {"direction": "down", "pixels": 600},
                    "is_navigation": False,
                    "description": "Scroll down to explore page"
                }
            ],
            "reasoning": "Scroll to see more content"
        }
    })
    
    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create plan with turn 1 flag
    empty_plan = ExecutionPlan(actions=[])
    result = await orchestrator.analyze_and_plan(
        goal="Explore the page",
        current_plan=empty_plan,
        page_analysis=sample_page_analysis,
        is_turn_1=True
    )
    
    # Verify valid action was kept
    assert len(result.actions) == 1
    assert result.actions[0].action_type == "scroll"


@pytest.mark.asyncio
async def test_non_turn_1_allows_multiple_actions(orchestrator, sample_page_analysis):
    """Test that non-turn 1 plans allow multiple actions."""
    # Mock API response with multiple actions
    mock_response = MagicMock()
    mock_response.error = None  # No error
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "plan": {
            "actions": [
                {
                    "action_type": "click",
                    "target_element_id": "username_input",
                    "parameters": {},
                    "is_navigation": False,
                    "description": "Click username field"
                },
                {
                    "action_type": "type",
                    "target_element_id": "username_input",
                    "parameters": {"text": "testuser"},
                    "is_navigation": False,
                    "description": "Type username"
                }
            ],
            "reasoning": "Fill in username"
        }
    })
    
    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create plan WITHOUT turn 1 flag
    empty_plan = ExecutionPlan(actions=[])
    result = await orchestrator.analyze_and_plan(
        goal="Log in",
        current_plan=empty_plan,
        page_analysis=sample_page_analysis,
        is_turn_1=False
    )
    
    # Verify multiple actions are allowed
    assert len(result.actions) == 2
    assert result.actions[0].action_type == "click"
    assert result.actions[1].action_type == "type"
