"""
GPT4Orchestrator for goal analysis and execution planning.

This module implements the orchestrator component of the adaptive agent system,
using GPT-4o to analyze task goals, maintain dynamic execution plans, and make
all strategic decisions.
"""

import json
import time
from typing import Optional, Union, List, Dict, Any

from openai import AsyncOpenAI
import opik

from agi_agents.models import (
    Action,
    ExecutionPlan,
    ExecutionError,
    PageAnalysis,
    ElementLocationFailure,
    ToolCall,
    InfoSeekingQuery,
)


class GPT4Orchestrator:
    """
    GPT-4o orchestrator that analyzes task goals and maintains dynamic execution plans.
    Acts as the central decision-maker, sending two types of queries:
    - Tool Calls: Execute actions via llm_call_step
    - Info Seeking Queries: Request element locations from Qwen
    
    This model is responsible for:
    1. Analyzing task goals and creating execution plans
    2. Revising plans when errors occur or elements cannot be located
    3. Deciding between direct action execution and element location queries
    4. Maintaining plan state and revision history
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        client: Optional[AsyncOpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the GPT-4o Orchestrator.
        
        Args:
            model: Model identifier for GPT-4o
            client: Optional pre-configured AsyncOpenAI client
            api_key: API key for OpenAI-compatible endpoint
            base_url: Base URL for OpenAI-compatible endpoint
        """
        self.model = model
        
        if client is not None:
            self.client = client
        else:
            # Use environment variable if available, otherwise use provided key
            api_key = api_key or "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"
            base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        
        # Initialize Opik client for tracing
        self.opik_client = opik.Opik(project_name="agi-llm")
        print(f"[OPIK] Initialized Opik client for orchestrator with project: agi-llm")

    def _validate_turn_1_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Validate that turn 1 plan meets constraints.
        
        Turn 1 constraints:
        - At most one exploratory action
        - No assumptions about page structure
        - Only navigation/observation actions (goto, wait, scroll)
        
        Args:
            plan: ExecutionPlan to validate
            
        Returns:
            Validated (and potentially corrected) ExecutionPlan
        """
        # Check if plan has more than one action
        if len(plan.actions) > 1:
            # Limit to first action only
            print(f"[ORCHESTRATOR] Turn 1 constraint violation: Plan has {len(plan.actions)} actions, limiting to 1")
            plan.actions = plan.actions[:1]
            plan.reasoning = (
                f"[Turn 1 Limited] {plan.reasoning} "
                "(Note: Multi-step plans only allowed after first observation)"
            )
        
        # Check if actions are valid for turn 1
        valid_turn_1_actions = {"goto", "wait", "scroll"}
        invalid_actions = []
        
        for action in plan.actions:
            if action.action_type not in valid_turn_1_actions:
                invalid_actions.append(action)
        
        # Remove invalid actions
        if invalid_actions:
            print(f"[ORCHESTRATOR] Turn 1 constraint violation: Removing {len(invalid_actions)} invalid actions")
            plan.actions = [a for a in plan.actions if a.action_type in valid_turn_1_actions]
            
            # If no valid actions remain, create a default exploratory action
            if not plan.actions:
                print("[ORCHESTRATOR] No valid turn 1 actions, creating default wait action")
                plan.actions = [
                    Action(
                        action_type="wait",
                        target_element_id="page",
                        parameters={"seconds": 2},
                        is_navigation=False,
                        description="Wait for page to load (turn 1 observation)"
                    )
                ]
                plan.reasoning = (
                    "Turn 1: Waiting to observe page structure before creating detailed plan"
                )
        
        return plan
    
    def _build_planning_prompt(
        self,
        goal: str,
        current_plan: ExecutionPlan,
        page_analysis: PageAnalysis,
        error_info: Optional[ExecutionError] = None,
        location_failure: Optional[ElementLocationFailure] = None,
        is_turn_1: bool = False,
        url: Optional[str] = None,
    ) -> str:
        """
        Build the prompt for plan generation or revision.
        
        Args:
            goal: Task goal to analyze
            current_plan: Current execution plan (may be empty)
            page_analysis: Latest page analysis from Qwen
            error_info: Error information if last action failed
            location_failure: Failure info if Qwen couldn't locate element
            is_turn_1: Whether this is the first turn (no prior observations)
            url: Optional URL for the task (useful for turn 1 navigation)
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add goal
        prompt_parts.append(f"# Task Goal\n{goal}\n")
        
        # Add URL if provided (especially useful for turn 1)
        if url:
            prompt_parts.append(f"# Task URL\n{url}\n")
        
        # Add page analysis
        prompt_parts.append("# Current Page Analysis")
        prompt_parts.append(f"Page Type: {page_analysis.page_type}")
        prompt_parts.append(f"Summary: {page_analysis.content_summary}\n")
        prompt_parts.append("Available Elements:")
        for elem in page_analysis.elements:
            field_info = f" ({elem.field_type})" if elem.field_type else ""
            prompt_parts.append(
                f"  - {elem.element_id}: {elem.element_type}{field_info} - '{elem.label}' at {elem.coordinates}"
            )
        prompt_parts.append("")
        
        # Add current plan status
        if current_plan.actions:
            prompt_parts.append("# Current Execution Plan")
            prompt_parts.append(f"Reasoning: {current_plan.reasoning}")
            prompt_parts.append(f"Revision Count: {current_plan.revision_count}")
            prompt_parts.append("Remaining Actions:")
            for i, action in enumerate(current_plan.actions, 1):
                prompt_parts.append(
                    f"  {i}. {action.action_type} on {action.target_element_id}: {action.description}"
                )
            prompt_parts.append("")
        else:
            prompt_parts.append("# Current Execution Plan\nNo plan exists - need to create initial plan.\n")
        
        # Add error information if present
        if error_info:
            prompt_parts.append("# Execution Error")
            prompt_parts.append(f"Failed Action: {error_info.action.description}")
            prompt_parts.append(f"Error Type: {error_info.error_type}")
            prompt_parts.append(f"Error Message: {error_info.error_message}")
            prompt_parts.append("You MUST revise the plan to address this error.\n")
        
        # Add location failure if present
        if location_failure:
            prompt_parts.append("# Element Location Failure")
            prompt_parts.append(f"Requested: {location_failure.element_description}")
            prompt_parts.append(f"Reason: {location_failure.reason}")
            prompt_parts.append("Available elements:")
            for elem in location_failure.available_elements:
                prompt_parts.append(f"  - {elem}")
            prompt_parts.append("You MUST revise the plan to use available elements.\n")
        
        # Add instructions
        prompt_parts.append("# Instructions")
        
        # Turn 1 specific instructions
        if is_turn_1:
            prompt_parts.append(
                "⚠️ TURN 1 CONSTRAINT: This is the first turn with NO prior page observations. "
                "You MUST create a plan with AT MOST ONE exploratory action. "
                "DO NOT assume knowledge of specific page elements or structure. "
                "Focus on navigation or simple observation actions only (e.g., goto, wait, scroll). "
                "Multi-step plans are ONLY allowed after observing the page in turn 1."
            )
            prompt_parts.append("")
        
        if error_info or location_failure or not current_plan.actions:
            if is_turn_1:
                prompt_parts.append(
                    "Create an initial exploratory plan with AT MOST ONE action. "
                    "Provide a JSON response with this structure:"
                )
            else:
                prompt_parts.append(
                    "Create or revise the execution plan to accomplish the goal. "
                    "Provide a JSON response with this structure:"
                )
        else:
            prompt_parts.append(
                "The current plan looks good. Continue with the next action from the plan. "
                "You can either:"
            )
            prompt_parts.append("1. Execute an action directly via tool call")
            prompt_parts.append("2. Request element location from vision model if needed")
            prompt_parts.append("\nProvide a JSON response with the appropriate structure.")
        
        prompt_parts.append("""
{
  "plan": {
    "actions": [
      {
        "action_type": "click|type|select|scroll|wait|goto",
        "target_element_id": "element_id_from_page_analysis",
        "parameters": {"key": "value"},
        "is_navigation": true|false,
        "description": "what this action does"
      }
    ],
    "reasoning": "brief 1-sentence explanation"
  }
}

## Action Types
- click: Click an element (parameters: {})
- type: Type text into an input (parameters: {"text": "content to type"})
- select: Select dropdown option (parameters: {"value": "option_value"})
- scroll: Scroll the page (parameters: {"direction": "up|down", "pixels": 600})
- wait: Wait for page to load (parameters: {"seconds": 2})
- goto: Navigate to URL (parameters: {"url": "https://example.com"})

## Important Guidelines
- Use element IDs from the page analysis
- For text inputs, use action_type="type"
- For dropdowns, use action_type="select"
- For date pickers, use action_type="type" with formatted date
- Mark actions that cause navigation with is_navigation=true
- Group same-page actions together for batching
- Be specific and actionable in descriptions
- If you need to locate an element not in the analysis, you can request it separately
- Keep reasoning brief (1 sentence max)""")
        
        # Add turn 1 specific guidelines
        if is_turn_1:
            prompt_parts.append("""

## TURN 1 CONSTRAINTS (CRITICAL)
- ⚠️ You have NOT observed the page yet - DO NOT reference specific elements
- ⚠️ Create a plan with AT MOST ONE exploratory action
- ⚠️ Valid turn 1 actions: goto (navigate to URL), wait (wait for page load), scroll (explore page)
- ⚠️ INVALID turn 1 actions: click, type, select (require knowing page structure)
- ⚠️ After turn 1 completes, you will see the page and can create detailed multi-step plans""")
        
        prompt_parts.append("\n\nRespond ONLY with the JSON object, no additional text.")
        
        return "\n".join(prompt_parts)

    async def analyze_and_plan(
        self,
        goal: str,
        current_plan: ExecutionPlan,
        page_analysis: PageAnalysis,
        error_info: Optional[ExecutionError] = None,
        location_failure: Optional[ElementLocationFailure] = None,
        is_turn_1: bool = False,
        url: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Analyze task goal and create/update execution plan.
        This runs in parallel with Qwen's screenshot analysis.
        
        Args:
            goal: Task goal to analyze
            current_plan: Current execution plan (may be empty)
            page_analysis: Latest page analysis from Qwen
            error_info: Error information if last action failed
            location_failure: Failure info if Qwen couldn't locate element
            is_turn_1: Whether this is the first turn (no prior observations)
            url: Optional URL for the task (useful for turn 1 navigation)
            
        Returns:
            Updated ExecutionPlan with actions to execute
            
        Raises:
            RuntimeError: If API call fails after retries
        """
        prompt = self._build_planning_prompt(
            goal=goal,
            current_plan=current_plan,
            page_analysis=page_analysis,
            error_info=error_info,
            location_failure=location_failure,
            is_turn_1=is_turn_1,
            url=url,
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert web automation orchestrator. "
                    "You analyze task goals and create detailed execution plans. "
                    "You make strategic decisions about what actions to take and when. "
                    "Always respond with valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        # Call API with retry logic and exponential backoff
        max_retries = 3
        last_error = None
        
        # Create Opik trace
        print("[OPIK] Creating trace for orchestrator call...")
        trace = self.opik_client.trace(
            project_name="agi-llm",
            name=f"gpt4o_orchestrator_plan",
            input={
                "goal": goal,
                "has_error": error_info is not None,
                "has_location_failure": location_failure is not None,
                "current_plan_actions": len(current_plan.actions),
                "revision_count": current_plan.revision_count,
            },
        )
        
        for attempt in range(max_retries):
            try:
                # Add exponential backoff delay before retry (except first attempt)
                if attempt > 0:
                    import asyncio
                    backoff_delay = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                    await asyncio.sleep(backoff_delay)
                
                print(f"[ORCHESTRATOR] Calling GPT-4o API (attempt {attempt + 1}/{max_retries})...")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=2048,
                    messages=messages,
                )
                
                # Check for API errors
                if hasattr(response, "error") and response.error:
                    error_msg = response.error.get("message", "Unknown error")
                    last_error = f"API error: {error_msg}"
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError(last_error)
                
                # Check for valid response
                if not response.choices or len(response.choices) == 0:
                    last_error = "Empty response from API"
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError(last_error)
                
                content = response.choices[0].message.content
                if not content:
                    last_error = "Empty content in response"
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError(last_error)
                
                # Parse JSON response
                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    json_str = content.strip()
                
                data = json.loads(json_str)
                
                # Build ExecutionPlan from response
                plan_data = data.get("plan", {})
                actions = []
                
                for action_data in plan_data.get("actions", []):
                    action = Action(
                        action_type=action_data["action_type"],
                        target_element_id=action_data["target_element_id"],
                        parameters=action_data.get("parameters", {}),
                        is_navigation=action_data.get("is_navigation", False),
                        description=action_data.get("description", ""),
                    )
                    actions.append(action)
                
                # Determine revision count
                revision_count = current_plan.revision_count
                if error_info or location_failure:
                    revision_count += 1
                
                execution_plan = ExecutionPlan(
                    actions=actions,
                    reasoning=plan_data.get("reasoning", ""),
                    created_at=time.time(),
                    revision_count=revision_count,
                )
                
                # Validate turn 1 constraints
                if is_turn_1:
                    execution_plan = self._validate_turn_1_plan(execution_plan)
                
                # Log to Opik
                try:
                    trace.span(
                        name=f"gpt4o_plan_generation",
                        type="llm",
                        input={
                            "messages": messages,
                            "model": self.model,
                        },
                        output={
                            "plan": {
                                "actions_count": len(execution_plan.actions),
                                "reasoning": execution_plan.reasoning,
                                "revision_count": execution_plan.revision_count,
                            }
                        },
                        metadata={
                            "attempt": attempt + 1,
                            "page_type": page_analysis.page_type,
                            "elements_count": len(page_analysis.elements),
                        }
                    )
                    print(f"[OPIK] Logged orchestrator call to trace")
                except Exception as e:
                    print(f"[OPIK] Failed to log orchestrator call: {e}")
                
                # Update trace with output
                try:
                    trace.update(
                        output={
                            "plan": {
                                "actions_count": len(execution_plan.actions),
                                "reasoning": execution_plan.reasoning,
                                "revision_count": execution_plan.revision_count,
                                "actions": [
                                    {
                                        "type": a.action_type,
                                        "target": a.target_element_id,
                                        "description": a.description,
                                    }
                                    for a in execution_plan.actions
                                ],
                            }
                        }
                    )
                    trace.end()
                    print(f"[OPIK] Trace ended successfully")
                except Exception as e:
                    print(f"[OPIK] Failed to end trace: {e}")
                
                # Fallback logic for failed action planning (empty plan with no reasoning)
                # Only apply fallback if we got an empty plan with no explanation
                if (not execution_plan.actions and 
                    not execution_plan.reasoning and 
                    not error_info and 
                    not location_failure):
                    # If we got an empty plan with no reasoning, create a minimal fallback
                    # This could happen if the orchestrator is confused or the page is unclear
                    return ExecutionPlan(
                        actions=[],
                        reasoning="Unable to generate actionable plan - may need more information or page context",
                        created_at=time.time(),
                        revision_count=revision_count,
                    )
                
                return execution_plan
                
            except json.JSONDecodeError as e:
                last_error = f"Failed to parse JSON response: {e}"
                if attempt < max_retries - 1:
                    continue
                # End trace on final failure
                try:
                    trace.update(output={"error": last_error})
                    trace.end()
                except Exception:
                    pass
                raise RuntimeError(last_error) from e
            except KeyError as e:
                last_error = f"Missing required field in response: {e}"
                if attempt < max_retries - 1:
                    continue
                # End trace on final failure
                try:
                    trace.update(output={"error": last_error})
                    trace.end()
                except Exception:
                    pass
                raise RuntimeError(last_error) from e
            except Exception as e:
                last_error = f"API call failed: {e}"
                if attempt < max_retries - 1:
                    continue
                # End trace on final failure
                try:
                    trace.update(output={"error": last_error})
                    trace.end()
                except Exception:
                    pass
                raise RuntimeError(last_error) from e
        
        # Fallback: Return empty ExecutionPlan instead of raising
        # This allows the system to continue and potentially recover
        # End trace with fallback info
        try:
            trace.update(
                output={
                    "fallback": True,
                    "error": last_error,
                    "retries": max_retries,
                }
            )
            trace.end()
            print(f"[OPIK] Trace ended with fallback")
        except Exception as e:
            print(f"[OPIK] Failed to end trace: {e}")
        
        return ExecutionPlan(
            actions=[],
            reasoning=f"Orchestrator failed after {max_retries} retries: {last_error}",
            created_at=time.time(),
            revision_count=current_plan.revision_count + 1 if error_info or location_failure else current_plan.revision_count,
        )

    def parse_output(self, response: str) -> Union[ToolCall, InfoSeekingQuery]:
        """
        Parse orchestrator output into tool calls or info seeking queries.
        
        Tool Call: Direct action execution via llm_call_step
        Info Seeking Query: Straightforward element location request to Qwen
                           (sent ONLY when there's a concrete plan to use it)
        
        Args:
            response: Raw response from GPT-4o
            
        Returns:
            Either ToolCall (execute via llm_call_step) or InfoSeekingQuery (ask Qwen)
            
        Raises:
            ValueError: If response cannot be parsed into either type
        """
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            # Check if it's a tool call
            if "tool_call" in data:
                tool_data = data["tool_call"]
                return ToolCall(
                    tool_name=tool_data.get("tool_name", "llm_call_step"),
                    action_type=tool_data["action_type"],
                    parameters=tool_data.get("parameters", {}),
                )
            
            # Check if it's an info seeking query
            if "info_query" in data:
                query_data = data["info_query"]
                return InfoSeekingQuery(
                    element_description=query_data["element_description"],
                    context=query_data.get("context", ""),
                )
            
            # If neither, raise error
            raise ValueError(
                "Response must contain either 'tool_call' or 'info_query' field"
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}") from e
        except KeyError as e:
            raise ValueError(f"Missing required field in response: {e}") from e
