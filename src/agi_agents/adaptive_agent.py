"""
AdaptiveAgent for plan-based web automation with dynamic error recovery.

This module implements the main agent class that orchestrates the adaptive
web automation system using parallel analysis from Qwen VLM and GPT-4o.
"""

import asyncio
from typing import Optional

from arena.agent import BaseAgent
from arena.browser import AgentBrowser
from arena.state import AgentState

from agi_agents.qwen.vision_model import QwenVisionModel
from agi_agents.orchestrator import GPT4Orchestrator
from agi_agents.batch_executor import ActionBatchExecutor
from agi_agents.metrics_tracker import MetricsTracker
from agi_agents.models import (
    ExecutionPlan,
    ExecutionError,
    PageAnalysis,
    ToolCall,
    InfoSeekingQuery,
    ElementLocation,
    ElementLocationFailure,
)


class AdaptiveAgent(BaseAgent):
    """
    Adaptive agent using plan-based execution with dynamic error recovery.
    
    Architecture:
    - Qwen VLM and GPT-4o work in parallel
    - GPT-4o is the central decision-maker
    - Two query types: Tool Calls (via llm_call_step) and Info Seeking Queries
    - Qwen reports failures back to GPT-4o for plan revision
    
    The agent implements a sophisticated workflow:
    1. Capture screenshot
    2. Parallel Analysis:
       - Qwen analyzes screenshot → PageAnalysis
       - GPT-4o analyzes goal → ExecutionPlan
    3. GPT-4o Decision:
       - Tool Call → Execute via llm_call_step
       - Info Seeking Query → Ask Qwen for element location
    4. Handle Info Seeking Query (if applicable):
       - Qwen returns coordinates OR failure
       - If failure → GPT-4o revises plan
    5. Execute action batch from plan
    6. Handle errors by revising plan
    7. Return updated state
    """
    
    def __init__(
        self,
        vision_model: QwenVisionModel,
        orchestrator_model: GPT4Orchestrator,
        metrics_tracker: MetricsTracker,
    ):
        """
        Initialize the AdaptiveAgent.
        
        Args:
            vision_model: QwenVisionModel for page analysis and element location
            orchestrator_model: GPT4Orchestrator for goal analysis and planning
            metrics_tracker: MetricsTracker for performance monitoring
        """
        self.vision_model = vision_model
        self.orchestrator_model = orchestrator_model
        self.metrics_tracker = metrics_tracker
        self.current_plan: ExecutionPlan = ExecutionPlan(actions=[])
        self.current_page_analysis: Optional[PageAnalysis] = None
        self.consecutive_revisions: int = 0
        self.max_consecutive_revisions: int = 3
    
    async def _capture_screenshot_with_retry(
        self,
        browser: AgentBrowser,
        max_retries: int = 3
    ) -> str:
        """
        Capture screenshot with retry logic.
        
        Implements retry logic for screenshot capture to handle transient
        browser errors or timing issues.
        
        Args:
            browser: AgentBrowser instance
            max_retries: Maximum number of retry attempts
            
        Returns:
            Base64-encoded screenshot string
            
        Raises:
            RuntimeError: If screenshot capture fails after all retries
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                screenshot = await browser.screenshot()
                return screenshot
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    # Wait a bit before retrying
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                # On final attempt, raise the error
                raise RuntimeError(
                    f"Screenshot capture failed after {max_retries} retries: {last_error}"
                ) from e
        
        # Should not reach here, but raise as fallback
        raise RuntimeError(
            f"Screenshot capture failed after {max_retries} retries: {last_error}"
        )
    
    async def _handle_tool_call(
        self,
        tool_call: ToolCall,
        browser: AgentBrowser,
        state: AgentState
    ) -> str:
        """
        Handle Tool Call by executing action via llm_call_step.
        
        This method executes actions directly through the browser automation
        primitives, similar to how the QwenAgent handles tool calls.
        
        Args:
            tool_call: ToolCall to execute
            browser: AgentBrowser instance
            state: Current agent state
            
        Returns:
            Result message from tool execution
        """
        # Import tool executor from qwen module
        from agi_agents.qwen.tools import QwenToolExecutor
        
        # Get viewport dimensions for coordinate scaling
        screenshot = await browser.screenshot()
        import base64
        import io
        from PIL import Image
        screenshot_bytes = base64.b64decode(screenshot)
        image = Image.open(io.BytesIO(screenshot_bytes))
        original_width, original_height = image.width, image.height
        
        # Create tool executor
        tool_executor = QwenToolExecutor(
            page=browser.page,
            browser=browser,
            state=state,
            original_width=original_width,
            original_height=original_height,
        )
        
        # Execute the tool
        try:
            result = await tool_executor.execute_tool(
                tool_call.action_type,
                tool_call.parameters
            )
            return result
        except Exception as e:
            return f"Tool execution failed: {str(e)}"
    
    async def _handle_info_seeking_query(
        self,
        query: InfoSeekingQuery,
        screenshot_bytes: bytes,
        state: AgentState
    ) -> str:
        """
        Handle Info Seeking Query by routing to Vision Model.
        
        This method sends straightforward element location requests to Qwen
        and returns either coordinates or failure information.
        
        Args:
            query: InfoSeekingQuery to process
            screenshot_bytes: Current screenshot
            state: Current agent state
            
        Returns:
            Result message with coordinates or failure information
        """
        if self.current_page_analysis is None:
            return "Error: No page analysis available"
        
        # Send query to vision model
        result = await self.vision_model.locate_element(
            screenshot=screenshot_bytes,
            element_description=query.element_description,
            page_analysis=self.current_page_analysis
        )
        
        # Check if it's a success or failure
        if isinstance(result, ElementLocation):
            # Success - return coordinates
            return (
                f"Element located: {result.element_id} at coordinates {result.coordinates} "
                f"(type: {result.element_type}, confidence: {result.confidence})"
            )
        elif isinstance(result, ElementLocationFailure):
            # Failure - return failure information
            available = ", ".join(result.available_elements[:5])  # Limit to first 5
            return (
                f"Element location failed: {result.reason}. "
                f"Available elements: {available}"
            )
        else:
            return "Error: Unexpected result type from vision model"
    
    async def _route_query(
        self,
        response: str,
        browser: AgentBrowser,
        state: AgentState,
        screenshot_bytes: bytes
    ) -> tuple[Optional[str], Optional[ElementLocationFailure]]:
        """
        Route orchestrator output to appropriate handler.
        
        Classifies the orchestrator output as either a ToolCall or InfoSeekingQuery
        and routes it to the appropriate handler.
        
        Args:
            response: Raw response from orchestrator
            browser: AgentBrowser instance
            state: Current agent state
            screenshot_bytes: Current screenshot
            
        Returns:
            Tuple of (result_message, location_failure)
            - result_message: Result from query execution
            - location_failure: ElementLocationFailure if query failed, None otherwise
        """
        try:
            # Parse the output
            query = self.orchestrator_model.parse_output(response)
            
            # Route based on query type
            if isinstance(query, ToolCall):
                # Handle tool call via llm_call_step
                result = await self._handle_tool_call(query, browser, state)
                return result, None
            
            elif isinstance(query, InfoSeekingQuery):
                # Handle info seeking query via vision model
                result = await self._handle_info_seeking_query(
                    query,
                    screenshot_bytes,
                    state
                )
                
                # Check if it was a failure
                vision_result = await self.vision_model.locate_element(
                    screenshot=screenshot_bytes,
                    element_description=query.element_description,
                    page_analysis=self.current_page_analysis or PageAnalysis(
                        elements=[],
                        page_type="unknown",
                        content_summary="",
                        timestamp=0.0
                    )
                )
                
                if isinstance(vision_result, ElementLocationFailure):
                    # Element location failed - trigger plan revision
                    await self._handle_location_failure(
                        vision_result,
                        browser,
                        state,
                        screenshot_bytes
                    )
                    return result, vision_result
                else:
                    return result, None
            
            else:
                return "Error: Unknown query type", None
                
        except ValueError as e:
            # Parse error - treat as invalid query
            return f"Error parsing orchestrator output: {e}", None
    
    async def _handle_location_failure(
        self,
        location_failure: ElementLocationFailure,
        browser: AgentBrowser,
        state: AgentState,
        screenshot_bytes: bytes
    ) -> None:
        """
        Handle element location failure by triggering plan revision.
        
        When Qwen cannot locate a requested element, this method triggers
        a plan revision with the failure information so GPT-4o can adjust
        the plan to use available elements.
        
        Args:
            location_failure: ElementLocationFailure from vision model
            browser: AgentBrowser instance
            state: Current agent state
            screenshot_bytes: Current screenshot
        """
        # Check revision limit
        if self.consecutive_revisions >= self.max_consecutive_revisions:
            # Fail the task after max consecutive revisions
            state.messages.append({
                "role": "system",
                "content": (
                    f"Task failed: Maximum consecutive plan revisions "
                    f"({self.max_consecutive_revisions}) reached. "
                    f"Unable to locate required elements."
                ),
                "task_failed": True,
                "revision_count": self.consecutive_revisions
            })
            state.finished = True
            return
        
        # Add failure information to state
        state.messages.append({
            "role": "system",
            "content": (
                f"Element location failed: {location_failure.reason}. "
                f"Requested: {location_failure.element_description}"
            ),
            "location_failure": True,
            "element_description": location_failure.element_description
        })
        
        # Call orchestrator to revise plan with location failure info
        self.metrics_tracker.record_orchestrator_call()
        self.metrics_tracker.record_plan_revision()
        
        revised_plan = await self.orchestrator_model.analyze_and_plan(
            goal=state.goal or "",
            current_plan=self.current_plan,
            page_analysis=self.current_page_analysis or PageAnalysis(
                elements=[],
                page_type="unknown",
                content_summary="",
                timestamp=0.0
            ),
            error_info=None,
            location_failure=location_failure,
            is_turn_1=False,  # Location failures are never turn 1
            url=state.url
        )
        
        # Update plan and increment revision counter
        self.current_plan = revised_plan
        self.consecutive_revisions += 1
        
        # Add revision message to state
        state.messages.append({
            "role": "system",
            "content": (
                f"Plan revised due to location failure (revision {self.consecutive_revisions}/"
                f"{self.max_consecutive_revisions}): {revised_plan.reasoning}"
            ),
            "revision_count": self.consecutive_revisions,
            "new_actions_count": len(revised_plan.actions)
        })
    
    async def step(self, browser: AgentBrowser, state: AgentState) -> AgentState:
        """
        Execute one agent step using adaptive planning.
        
        This method is async to integrate with the existing TaskExecution workflow.
        
        Args:
            browser: AgentBrowser instance for web automation
            state: Current agent state
            
        Returns:
            Updated agent state after step execution
        """
        # Execute the async step
        result_state = await self._async_step(browser, state)
        
        # Stop metrics tracking if task is done
        if result_state.finished:
            self.metrics_tracker.stop()
        
        return result_state
    
    def get_metrics_summary(self) -> "MetricsSummary":
        """
        Get comprehensive metrics summary for the execution.
        
        Returns:
            MetricsSummary with all tracked metrics and statistics
        """
        return self.metrics_tracker.get_summary()
    
    async def _async_step(self, browser: AgentBrowser, state: AgentState) -> AgentState:
        """
        Execute one agent step using adaptive planning (async implementation).
        
        Workflow:
        1. Capture screenshot
        2. Parallel Analysis:
           - Qwen analyzes screenshot → PageAnalysis
           - GPT-4o analyzes goal → Update/Create Plan
        3. GPT-4o Decision Making:
           - Tool Call → Execute via llm_call_step
           - Info Query → Ask Qwen for element location
        4. Handle Info Seeking Query (if applicable):
           - Qwen returns coordinates OR failure
           - If failure → GPT-4o revises plan
        5. Execute action batch from plan
        6. If Error or Qwen Fails → GPT-4o Revises Plan
        7. If Plan Empty → GPT-4o Creates New Plan
        8. Return updated state
        
        Args:
            browser: AgentBrowser instance for web automation
            state: Current agent state
            
        Returns:
            Updated agent state after step execution
        """
        # Start metrics tracking if this is the first step
        if state.step == 0:
            self.metrics_tracker.start()
        
        # Step 1: Capture screenshot with retry logic
        screenshot = await self._capture_screenshot_with_retry(browser)
        self.metrics_tracker.record_screenshot()
        
        # Convert Base64Image to bytes for vision model
        import base64
        screenshot_bytes = base64.b64decode(screenshot)
        
        # Step 2: Parallel Analysis
        # Detect if this is turn 1 (no prior observations)
        # Turn 1 is when step == 0 (first step) and we have no page analysis yet
        is_turn_1 = (state.step == 0 and self.current_page_analysis is None)
        
        # Performance Optimization: Skip orchestrator call if plan has valid actions
        # This reduces API calls when we already have a good plan to execute
        should_call_orchestrator = (
            state.step == 0 or 
            not self.current_plan.actions or
            self.consecutive_revisions > 0  # Always call after errors
        )
        
        if should_call_orchestrator:
            # Launch both models in parallel without blocking
            print(f"[ADAPTIVE_AGENT] Launching parallel analysis (is_turn_1={is_turn_1})...")
            print("[ADAPTIVE_AGENT] Starting vision model task...")
            vision_task = asyncio.create_task(
                self.vision_model.analyze_page(
                    screenshot=screenshot_bytes,
                    history=state.messages
                )
            )
            
            print("[ADAPTIVE_AGENT] Starting orchestrator task...")
            orchestrator_task = asyncio.create_task(
                self.orchestrator_model.analyze_and_plan(
                    goal=state.goal or "",
                    current_plan=self.current_plan,
                    page_analysis=self.current_page_analysis or PageAnalysis(
                        elements=[],
                        page_type="unknown",
                        content_summary="",
                        timestamp=0.0
                    ),
                    error_info=None,
                    location_failure=None,
                    is_turn_1=is_turn_1,
                    url=state.url
                )
            )
            
            # Record that we're calling both models
            self.metrics_tracker.record_vision_call()
            self.metrics_tracker.record_orchestrator_call()
            print("[ADAPTIVE_AGENT] Both tasks created, waiting for completion...")
            
            # Wait for both to complete with timeout
            print("[ADAPTIVE_AGENT] Waiting for parallel analysis to complete...")
            try:
                page_analysis, execution_plan = await asyncio.wait_for(
                    asyncio.gather(vision_task, orchestrator_task),
                    timeout=120.0  # 2 minute timeout
                )
                print("[ADAPTIVE_AGENT] Parallel analysis completed successfully")
            except asyncio.TimeoutError:
                print("[ADAPTIVE_AGENT] WARNING: Parallel analysis timed out after 120s")
                # Try to get partial results
                if vision_task.done():
                    page_analysis = vision_task.result()
                    print("[ADAPTIVE_AGENT] Vision model completed")
                else:
                    print("[ADAPTIVE_AGENT] Vision model still running, using empty analysis")
                    page_analysis = PageAnalysis(
                        elements=[],
                        page_type="unknown",
                        content_summary="Vision model timed out",
                        timestamp=0.0
                    )
                
                if orchestrator_task.done():
                    execution_plan = orchestrator_task.result()
                    print("[ADAPTIVE_AGENT] Orchestrator completed")
                else:
                    print("[ADAPTIVE_AGENT] Orchestrator still running, using empty plan")
                    execution_plan = ExecutionPlan(actions=[], reasoning="Orchestrator timed out", created_at=0.0)
            
            # Store results
            self.current_page_analysis = page_analysis
            self.current_plan = execution_plan
        else:
            # Performance Optimization: Only call vision model when plan exists
            # This ensures vision model is called exactly once per screenshot
            self.metrics_tracker.record_vision_call()
            page_analysis = await self.vision_model.analyze_page(
                screenshot=screenshot_bytes,
                history=state.messages
            )
            self.current_page_analysis = page_analysis
            # Keep existing plan
        
        # Update state with new information
        state.step += 1
        
        # Add page analysis to state messages for context
        state.messages.append({
            "role": "system",
            "content": f"Page Analysis: {page_analysis.content_summary}",
            "page_type": page_analysis.page_type,
            "elements_count": len(page_analysis.elements)
        })
        
        # Add execution plan to state messages (if orchestrator was called)
        if should_call_orchestrator:
            state.messages.append({
                "role": "assistant",
                "content": f"Execution Plan: {self.current_plan.reasoning}",
                "actions_count": len(self.current_plan.actions),
                "revision_count": self.current_plan.revision_count
            })
        
        # Step 4: Execute action batch from plan
        if self.current_plan.actions:
            # Create batch executor
            batch_executor = ActionBatchExecutor(browser)
            
            # Extract next batch of actions
            action_batch = batch_executor.create_batch(self.current_plan)
            
            if action_batch:
                # Execute the batch
                result = await batch_executor.execute_batch(
                    actions=action_batch,
                    page_analysis=self.current_page_analysis
                )
                
                # Record action execution metrics
                self.metrics_tracker.record_actions(
                    total=len(action_batch),
                    successful=result.actions_completed if result.success else result.actions_completed
                )
                
                # Handle batch execution result
                if result.success:
                    # Batch completed successfully
                    # Conditional screenshot capture: only if navigation occurred
                    if result.navigation_occurred:
                        screenshot = await self._capture_screenshot_with_retry(browser)
                        self.metrics_tracker.record_screenshot()
                        screenshot_bytes = base64.b64decode(screenshot)
                    
                    # Remove completed actions from plan
                    self.current_plan.actions = self.current_plan.actions[result.actions_completed:]
                    
                    # Add success message to state
                    state.messages.append({
                        "role": "system",
                        "content": f"Successfully executed {result.actions_completed} actions. Navigation occurred: {result.navigation_occurred}",
                        "actions_completed": result.actions_completed,
                        "navigation_occurred": result.navigation_occurred
                    })
                    
                    # Reset consecutive revision counter on success
                    self.consecutive_revisions = 0
                    
                else:
                    # Batch execution failed - capture screenshot
                    screenshot = await self._capture_screenshot_with_retry(browser)
                    self.metrics_tracker.record_screenshot()
                    screenshot_bytes = base64.b64decode(screenshot)
                    
                    # Partial batch execution recovery: Remove successfully completed actions
                    # This preserves progress and allows the orchestrator to revise only the remaining plan
                    if result.actions_completed > 0:
                        # Some actions succeeded before the failure
                        self.current_plan.actions = self.current_plan.actions[result.actions_completed:]
                        state.messages.append({
                            "role": "system",
                            "content": f"Partial batch execution: {result.actions_completed} actions completed successfully before failure",
                            "actions_completed": result.actions_completed,
                            "remaining_actions": len(self.current_plan.actions)
                        })
                    
                    # Add error information to state
                    if result.error:
                        # Provide detailed error context for better recovery
                        error_context = {
                            "role": "system",
                            "content": f"Action execution failed: {result.error.error_message}",
                            "error_type": result.error.error_type,
                            "failed_action": result.error.action.description,
                            "failed_action_type": result.error.action.action_type,
                            "target_element": result.error.action.target_element_id,
                            "actions_completed": result.actions_completed,
                            "partial_success": result.actions_completed > 0
                        }
                        state.messages.append(error_context)
                        
                        # Check revision limit
                        if self.consecutive_revisions >= self.max_consecutive_revisions:
                            # Fail the task after max consecutive revisions
                            state.messages.append({
                                "role": "system",
                                "content": (
                                    f"Task failed: Maximum consecutive plan revisions "
                                    f"({self.max_consecutive_revisions}) reached. "
                                    f"Unable to recover from errors."
                                ),
                                "task_failed": True,
                                "revision_count": self.consecutive_revisions
                            })
                            state.finished = True
                            return state
                        
                        # Trigger plan revision with error information
                        # First, update page analysis with current screenshot
                        self.metrics_tracker.record_vision_call()
                        page_analysis = await self.vision_model.analyze_page(
                            screenshot=screenshot_bytes,
                            history=state.messages
                        )
                        self.current_page_analysis = page_analysis
                        
                        # Call orchestrator to revise plan
                        self.metrics_tracker.record_orchestrator_call()
                        self.metrics_tracker.record_plan_revision()
                        
                        revised_plan = await self.orchestrator_model.analyze_and_plan(
                            goal=state.goal or "",
                            current_plan=self.current_plan,
                            page_analysis=page_analysis,
                            error_info=result.error,
                            location_failure=None,
                            is_turn_1=False,  # Revisions are never turn 1
                            url=state.url
                        )
                        
                        # Update plan and increment revision counter
                        self.current_plan = revised_plan
                        self.consecutive_revisions += 1
                        
                        # Add revision message to state
                        state.messages.append({
                            "role": "system",
                            "content": (
                                f"Plan revised (revision {self.consecutive_revisions}/"
                                f"{self.max_consecutive_revisions}): {revised_plan.reasoning}"
                            ),
                            "revision_count": self.consecutive_revisions,
                            "new_actions_count": len(revised_plan.actions)
                        })
        
        return state

