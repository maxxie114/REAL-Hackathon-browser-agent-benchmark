"""
Action batch executor for the adaptive web automation agent.

This module implements batched execution of actions from the execution plan,
executing consecutive same-page actions without intermediate screenshots for
improved performance.
"""

from typing import List, Optional
from patchright.async_api import Page

from arena.browser import AgentBrowser
from agi_agents.models import (
    Action,
    ExecutionPlan,
    ExecutionResult,
    ExecutionError,
    PageAnalysis,
    PageElement,
)


class ActionBatchExecutor:
    """
    Executes action batches with error handling.
    
    Extracts consecutive same-page actions from the execution plan and
    executes them sequentially without intermediate screenshots. Stops
    immediately on first error.
    """

    # Element type to action type mapping
    ELEMENT_TYPE_TO_ACTION = {
        "input": "type",
        "text": "type",
        "email": "type",
        "password": "type",
        "search": "type",
        "tel": "type",
        "url": "type",
        "number": "type",
        "textarea": "type",
        "select": "select",
        "dropdown": "select",
        "date": "date_input",
        "datetime": "date_input",
        "datetime-local": "date_input",
        "time": "date_input",
        "month": "date_input",
        "week": "date_input",
    }

    def __init__(self, browser: AgentBrowser):
        """
        Initialize the action batch executor.
        
        Args:
            browser: AgentBrowser instance for executing actions
        """
        self.browser = browser

    def select_action_type(self, element: PageElement) -> str:
        """
        Select the appropriate action type based on element type and field type.
        
        Implements intelligent action selection:
        - Text inputs → "type" action
        - Dropdowns/selects → "select" action
        - Date pickers → "date_input" action
        
        Args:
            element: PageElement to determine action type for
            
        Returns:
            Action type string ("type", "select", "date_input", or "click" as default)
        """
        # Check field_type first (more specific for input elements)
        if element.field_type:
            action_type = self.ELEMENT_TYPE_TO_ACTION.get(element.field_type.lower())
            if action_type:
                return action_type
        
        # Fall back to element_type
        element_type_lower = element.element_type.lower()
        action_type = self.ELEMENT_TYPE_TO_ACTION.get(element_type_lower)
        if action_type:
            return action_type
        
        # Default to click for buttons, links, and unknown elements
        return "click"

    def create_batch(self, plan: ExecutionPlan) -> List[Action]:
        """
        Extract next batch of same-page actions from plan.
        
        A batch consists of consecutive actions that don't cause navigation.
        The batch stops at the first navigation action (which is included)
        or when all actions are exhausted.
        
        Args:
            plan: Current execution plan
            
        Returns:
            List of actions that can be batched together
        """
        if not plan.actions:
            return []
        
        batch: List[Action] = []
        
        for action in plan.actions:
            batch.append(action)
            # If this action causes navigation, stop the batch here
            if action.is_navigation:
                break
        
        return batch

    async def execute_batch(
        self,
        actions: List[Action],
        page_analysis: PageAnalysis
    ) -> ExecutionResult:
        """
        Execute a batch of actions sequentially.
        
        Actions are executed in order without intermediate screenshots.
        Execution stops immediately on first error.
        
        Args:
            actions: List of actions to execute
            page_analysis: Current page analysis for element lookup
            
        Returns:
            ExecutionResult with success status and error details if failed
        """
        if not actions:
            return ExecutionResult(
                success=True,
                actions_completed=0,
                navigation_occurred=False
            )
        
        # Create element lookup map
        element_map = {elem.element_id: elem for elem in page_analysis.elements}
        
        actions_completed = 0
        navigation_occurred = False
        
        for action in actions:
            try:
                # Execute the action
                await self._execute_action(action, element_map)
                actions_completed += 1
                
                # Track if navigation occurred
                if action.is_navigation:
                    navigation_occurred = True
                
            except Exception as e:
                # Stop batch execution on first error
                error = ExecutionError(
                    action=action,
                    error_type=self._classify_error(e),
                    error_message=str(e),
                    screenshot_path=None  # Screenshot will be captured by caller
                )
                
                return ExecutionResult(
                    success=False,
                    actions_completed=actions_completed,
                    error=error,
                    navigation_occurred=navigation_occurred
                )
        
        # All actions completed successfully
        return ExecutionResult(
            success=True,
            actions_completed=actions_completed,
            navigation_occurred=navigation_occurred
        )

    async def _execute_action(
        self,
        action: Action,
        element_map: dict[str, PageElement]
    ) -> None:
        """
        Execute a single action using browser primitives.
        
        Validates action type against element type and raises an error
        if there's a mismatch (e.g., trying to type into a dropdown).
        
        Args:
            action: Action to execute
            element_map: Map of element IDs to PageElement objects
            
        Raises:
            Exception: If action execution fails
        """
        page = self.browser.page
        
        # Get target element if specified
        target_element = None
        if action.target_element_id:
            target_element = element_map.get(action.target_element_id)
            if not target_element:
                raise ValueError(
                    f"Element not found: {action.target_element_id}"
                )
            
            # Validate action type matches element type
            if action.action_type in ["type", "select", "date_input"]:
                expected_action = self.select_action_type(target_element)
                if action.action_type != expected_action:
                    raise ValueError(
                        f"Action type mismatch: trying to use '{action.action_type}' "
                        f"on element type '{target_element.element_type}' "
                        f"(field_type: {target_element.field_type}). "
                        f"Expected action type: '{expected_action}'"
                    )
        
        # Execute based on action type
        if action.action_type == "click":
            await self._execute_click(page, target_element, action.parameters)
        
        elif action.action_type == "type":
            await self._execute_type(page, target_element, action.parameters)
        
        elif action.action_type == "select":
            await self._execute_select(page, target_element, action.parameters)
        
        elif action.action_type == "date_input":
            await self._execute_date_input(page, target_element, action.parameters)
        
        elif action.action_type == "scroll":
            await self._execute_scroll(page, action.parameters)
        
        elif action.action_type == "wait":
            await self._execute_wait(page, action.parameters)
        
        elif action.action_type == "goto":
            await self._execute_goto(page, action.parameters)
        
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    async def _execute_click(
        self,
        page: Page,
        element: Optional[PageElement],
        parameters: dict
    ) -> None:
        """Execute a click action."""
        if not element:
            raise ValueError("Click action requires target element")
        
        x, y = element.coordinates
        await page.mouse.click(x, y)

    async def _execute_type(
        self,
        page: Page,
        element: Optional[PageElement],
        parameters: dict
    ) -> None:
        """Execute a type action."""
        if not element:
            raise ValueError("Type action requires target element")
        
        # Click to focus the element first
        x, y = element.coordinates
        await page.mouse.click(x, y)
        
        # Get text to type
        text = parameters.get("text", "")
        if not text:
            raise ValueError("Type action requires 'text' parameter")
        
        # Type the text
        delay = parameters.get("delay", 50)
        await page.keyboard.type(text, delay=delay)

    async def _execute_select(
        self,
        page: Page,
        element: Optional[PageElement],
        parameters: dict
    ) -> None:
        """Execute a select action for dropdown elements."""
        if not element:
            raise ValueError("Select action requires target element")
        
        # Get value to select
        value = parameters.get("value", "")
        if value is None:
            raise ValueError("Select action requires 'value' parameter")
        
        # Click to open the dropdown
        x, y = element.coordinates
        await page.mouse.click(x, y)
        
        # Wait a bit for dropdown to open
        await page.wait_for_timeout(500)
        
        # Try to find and select the option
        # This is a simplified implementation - may need enhancement
        # for different dropdown types
        try:
            # Try native select element first
            selects = await page.query_selector_all("select")
            for select in selects:
                try:
                    await select.select_option(value=value)
                    return
                except Exception:
                    continue
            
            # If no native select worked, try clicking on option text
            # This handles custom dropdowns
            option_text = str(value)
            option = await page.query_selector(
                f"text={option_text}"
            )
            if option:
                await option.click()
            else:
                raise ValueError(f"Could not find option with value: {value}")
                
        except Exception as e:
            raise ValueError(f"Failed to select option: {e}")

    async def _execute_date_input(
        self,
        page: Page,
        element: Optional[PageElement],
        parameters: dict
    ) -> None:
        """Execute a date input action for date picker elements."""
        if not element:
            raise ValueError("Date input action requires target element")
        
        # Get date value to input
        date_value = parameters.get("value", "")
        if not date_value:
            raise ValueError("Date input action requires 'value' parameter")
        
        # Click to focus the element
        x, y = element.coordinates
        await page.mouse.click(x, y)
        
        # Wait a bit for any date picker UI to appear
        await page.wait_for_timeout(300)
        
        # Try to input the date value
        # For native HTML5 date inputs, we can type directly
        try:
            # Clear any existing value first
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Backspace")
            
            # Type the date value
            # Format should be appropriate for the input type (e.g., YYYY-MM-DD for date)
            await page.keyboard.type(str(date_value), delay=50)
            
            # Press Enter to confirm (some date pickers need this)
            await page.keyboard.press("Enter")
            
        except Exception as e:
            raise ValueError(f"Failed to input date: {e}")

    async def _execute_scroll(
        self,
        page: Page,
        parameters: dict
    ) -> None:
        """Execute a scroll action."""
        direction = parameters.get("direction", "down")
        pixels = parameters.get("pixels", 300)
        
        # Get viewport center for scrolling
        viewport_size = await page.evaluate(
            "({width: window.innerWidth, height: window.innerHeight})"
        )
        scroll_x = viewport_size["width"] / 2
        scroll_y = viewport_size["height"] / 2
        
        # Move mouse to center
        await page.mouse.move(scroll_x, scroll_y)
        
        # Scroll in the specified direction
        if direction == "down":
            await page.mouse.wheel(0, pixels)
        elif direction == "up":
            await page.mouse.wheel(0, -pixels)
        elif direction == "right":
            await page.mouse.wheel(pixels, 0)
        elif direction == "left":
            await page.mouse.wheel(-pixels, 0)
        else:
            raise ValueError(f"Unknown scroll direction: {direction}")

    async def _execute_wait(
        self,
        page: Page,
        parameters: dict
    ) -> None:
        """Execute a wait action."""
        duration = parameters.get("duration", 1000)
        await page.wait_for_timeout(duration)

    async def _execute_goto(
        self,
        page: Page,
        parameters: dict
    ) -> None:
        """Execute a goto (navigation) action."""
        url = parameters.get("url", "")
        if not url:
            raise ValueError("Goto action requires 'url' parameter")
        
        await page.goto(url.strip())

    def _classify_error(self, error: Exception) -> str:
        """
        Classify an error into a category.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error type string
        """
        error_str = str(error).lower()
        
        if "not found" in error_str or "element not found" in error_str:
            return "element_not_found"
        elif "timeout" in error_str:
            return "timeout"
        elif "type mismatch" in error_str or "field type" in error_str:
            return "type_mismatch"
        elif "navigation" in error_str:
            return "navigation_error"
        else:
            return "unknown_error"
