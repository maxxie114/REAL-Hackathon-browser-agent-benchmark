import asyncio
import copy
import datetime
import base64
import io
import json
import re
from typing import List, Dict, Any, Tuple

from patchright.async_api import Page
import pytz
from openai import AsyncOpenAI
from PIL import Image
import opik
from opik import Attachment

from arena import BaseAgent, AgentBrowser, AgentState
from agi_agents.prompts import QWEN_AGENT
from .tools import QwenToolExecutor


def _pil_to_data_url(image: Image.Image) -> str:
    """Convert a PIL image to a data URL suitable for OpenAI-compatible image_url content."""
    buf = io.BytesIO()
    fmt = "JPEG"
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format=fmt, quality=80)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class QwenAgent(BaseAgent):
    """Browser automation agent using Qwen with text-based tool calling via DashScope."""

    def __init__(
        self,
        # model: str = "qwen3-vl-plus",
+       model: str = "openai/gpt-4o",
        date_mode: str = "current",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.date_mode = date_mode
        assert date_mode in ["fixed", "current"]

        # Use environment variable if available, otherwise use hardcoded key
        api_key = api_key or "your-openrouter-api-key"
        base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Initialize Opik client for tracing
        self.opik_client = opik.Opik(project_name="agi")
        print(f"[OPIK] Initialized Opik client with project: agi")

        # Min/max pixels for Qwen vision model
        self.min_pixels = 64 * 32 * 32
        self.max_pixels = 9800 * 32 * 32
        # Number of recent images to include (including current)
        self.visual_history_length = 4

    async def _get_expanded_select(self, page: Page):
        """Return first <select> element-handle that is AX-expanded, else None."""
        try:
            selects = await page.query_selector_all("select")
            for sel in selects:
                try:
                    ax = await page.accessibility.snapshot(root=sel)
                    if ax and ax.get("expanded"):
                        return sel
                except Exception:
                    continue  # element went stale
        except Exception:
            pass  # navigation race
        return None

    async def _check_dropdown_options(self, page):
        """Check for open dropdowns and return their options."""
        sel = await self._get_expanded_select(page)
        if sel:
            try:
                vals = await sel.evaluate(
                    "s => Array.from(s.options).map(o => o.value)"
                )
                if vals:
                    # Format for agent: show values clearly
                    return [f"'{val}'" if val else "''" for val in vals]
            except Exception:
                pass  # element went stale or other error
        return None

    async def _take_screenshot_cdp(self, browser: AgentBrowser) -> bytes:
        """Take screenshot instantly using CDP."""
        cdp = await browser.context.new_cdp_session(browser.page)
        result = await cdp.send(
            "Page.captureScreenshot", {"format": "jpeg", "quality": 80}
        )
        return base64.b64decode(result["data"])

    def _screenshot_to_data_url(self, screenshot_bytes: bytes) -> str:
        """Convert screenshot bytes to data URL without resizing."""
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    async def build_messages(
        self, state: AgentState, screenshot: bytes
    ) -> List[Dict[str, Any]]:
        """Build message list for OpenAI API from state."""
        # Build system message
        if self.date_mode == "current":
            pacific_tz = pytz.timezone("US/Pacific")
            current_date = datetime.datetime.now(pacific_tz).strftime("%Y-%m-%d")
            current_date = f"y-m-d: {current_date}"
        else:
            current_date = "Year 2024"

        system_message = {
            "role": "system",
            "content": QWEN_AGENT.format(date=current_date),
        }

        # If no messages yet, create first user message with goal
        if not state.messages:
            state.messages.append(
                {
                    "role": "user",
                    "content": f"## Task Goal\n{state.goal}",
                }
            )

        # Copy messages and then prune older image parts, keeping only the last N images
        messages = [system_message] + copy.deepcopy(state.messages)

        # Gather all image content parts across messages (skip system at index 0)
        image_positions = []
        for mi, msg in enumerate(messages[1:], start=1):
            content = msg.get("content")
            if isinstance(content, list):
                for pi, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_positions.append((mi, pi))

        # Decide which image parts to keep: last self.visual_history_length images
        keep_set = set(
            image_positions[-self.visual_history_length :]
            if self.visual_history_length > 0
            else []
        )

        # Remove image parts not in keep_set; drop empty messages that become [] content
        msgs_to_remove = []
        for mi, msg in enumerate(messages):
            content = msg.get("content")
            if isinstance(content, list):
                new_parts = []
                for pi, part in enumerate(content):
                    is_image = (
                        isinstance(part, dict) and part.get("type") == "image_url"
                    )
                    if not is_image:
                        new_parts.append(part)
                    else:
                        if (mi, pi) in keep_set:
                            part.setdefault("min_pixels", self.min_pixels)
                            part.setdefault("max_pixels", self.max_pixels)
                            new_parts.append(part)
                if len(new_parts) == 0:
                    msgs_to_remove.append(mi)
                else:
                    msg["content"] = new_parts

        # Remove messages that no longer have any content
        for idx in reversed(msgs_to_remove):
            del messages[idx]

        return messages

    def _parse_tool_calls(self, content: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Parse all tool calls from text content.

        Handles formats like:
        - click({"point_2d": [158, 100]})
        - type({"content": "hello"})

        Returns list of (tool_name, tool_args) tuples.
        """
        if not content:
            return []

        pattern = r"(\w+)\s*\(\s*(\{.*?\})\s*\)"
        tool_calls: List[Tuple[str, Dict[str, Any]]] = []

        for match in re.finditer(pattern, content, flags=re.DOTALL):
            func_name = match.group(1)
            json_str = match.group(2)
            try:
                arguments = json.loads(json_str)
            except json.JSONDecodeError:
                continue
            tool_calls.append((func_name, arguments))

        return tool_calls

    async def step(self, browser: AgentBrowser, state: AgentState) -> AgentState:
        """Execute one agent step using Qwen's native tool calling."""
        print(f"\n{'='*60}")
        print(f"[STEP] Starting new agent step")
        print(f"[STEP] Model: {self.model}")
        print(f"[STEP] Current state finished: {state.finished}")
        print(f"{'='*60}\n")

        state.model = self.model

        # Get viewport dimensions for coordinate scaling
        print("[SCREENSHOT] Taking screenshot via CDP...")
        screenshot = await self._take_screenshot_cdp(browser)
        print(f"[SCREENSHOT] Screenshot captured: {len(screenshot)} bytes")
        # Always derive dimensions from the screenshot so scaling matches real pixels
        image = Image.open(io.BytesIO(screenshot))
        original_width, original_height = image.width, image.height
        print(f"[IMAGE] Dimensions: {original_width}x{original_height}")

        # Ensure first turn includes task goal before image
        if not state.messages:
            print(f"[MESSAGES] First turn - adding task goal")
            print(f"[GOAL] {state.goal}")
            state.messages.append(
                {"role": "user", "content": f"## Task Goal\n{state.goal}"}
            )

        # Append current screenshot message to state to interleave with dialogue

        data_url = self._screenshot_to_data_url(screenshot)

        state.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                ],
            }
        )

        # Build messages (will filter to the last 4 images)
        messages = await self.build_messages(state, screenshot)
        print(f"[MESSAGES] Built {len(messages)} messages for API call")
        print(f"[MESSAGES] Message history length: {len(state.messages)}")

        # Call API with retry logic and Opik tracing
        print("[API] Calling Qwen API...")
        print("[OPIK] Creating trace for API call...")

        # Create Opik trace
        trace = self.opik_client.trace(
            name=f"qwen_step_{state.step}",
            input={"goal": state.goal, "step": state.step},
            metadata={"model": self.model, "task_goal": state.goal}
        )

        # Save screenshot to a temporary file for Opik attachment
        # Don't delete immediately - let Opik upload it first
        screenshot_path = None
        try:
            import tempfile
            screenshot_image = Image.open(io.BytesIO(screenshot))
            # Use delete=False and don't manually delete - let OS clean up temp files
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as tmp_file:
                screenshot_path = tmp_file.name
                screenshot_image.save(tmp_file, format='JPEG', quality=80)
            print(f"[OPIK] Saved screenshot to temp file: {screenshot_path}")
        except Exception as e:
            print(f"[OPIK] Failed to save screenshot: {e}")

        max_retries = 100
        response = None
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=1024,
                    messages=messages,
                )

                # Check if response has an error field (OpenRouter API errors)
                if hasattr(response, 'error') and response.error:
                    error_msg = response.error.get('message', 'Unknown error')
                    error_code = response.error.get('code', 'N/A')
                    last_error = f"API error {error_code}: {error_msg}"
                    print(f"[API] Attempt {attempt + 1} failed with error: {last_error}")
                    if attempt == max_retries:
                        raise RuntimeError(f"Qwen API call failed after {max_retries} retries: {last_error}")
                    continue

                # Check if response has valid choices
                if not response.choices or len(response.choices) == 0:
                    last_error = "Empty choices in response"
                    print(f"[API] Attempt {attempt + 1} failed: {last_error}")
                    if attempt == max_retries:
                        raise RuntimeError(f"Qwen API returned empty choices after {max_retries} retries")
                    continue

                print(f"[API] Success on attempt {attempt + 1}")
                break
            except Exception as e:
                last_error = str(e)
                print(f"[API] Attempt {attempt + 1} failed with exception: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Qwen API call failed after {max_retries} retries: {e}") from e

        if response is None:
            raise RuntimeError(f"Qwen API call failed without response. Last error: {last_error}")
        message = response.choices[0].message

        print(f"\n[RESPONSE] Assistant message:")
        print(f"{message.content}")
        print()

        # Log LLM call as a span with screenshot attachment
        try:
            # Prepare attachments
            attachments = []
            if screenshot_path:
                attachments.append(
                    Attachment(
                        data=screenshot_path,
                        content_type="image/jpeg",
                        name=f"screenshot_step_{state.step}.jpg"
                    )
                )

            # Create LLM span with screenshot
            trace.span(
                name=f"llm_call_step_{state.step}",
                type="llm",
                input={"messages": str(messages)[:1000], "goal": state.goal},  # Truncate for readability
                output={"content": message.content},
                model=self.model,
                metadata={
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "step": state.step
                },
                attachments=attachments
            )
            print(f"[OPIK] Logged LLM call with screenshot to trace")
        except Exception as e:
            print(f"[OPIK] Failed to log LLM call: {e}")

        # Save assistant message to state (just content, no tool parsing)
        state.messages.append({"role": "assistant", "content": message.content or ""})

        # Parse and execute tool calls
        tool_calls = self._parse_tool_calls(message.content)
        print(f"[TOOLS] Parsed {len(tool_calls)} tool calls")

        if tool_calls:
            print(f"[TOOLS] Executing {len(tool_calls)} tool calls...")
            tool_executor = QwenToolExecutor(
                page=browser.page,
                browser=browser,
                state=state,
                original_width=original_width,
                original_height=original_height,
            )

            tool_results: List[str] = []

            for i, (tool_name, tool_args) in enumerate(tool_calls, 1):
                print(f"[TOOL {i}] Executing: {tool_name}({tool_args})")
                try:
                    result_text = await tool_executor.execute_tool(tool_name, tool_args)
                    print(f"[TOOL {i}] Result: {result_text}")
                    if tool_name == "finished":
                        state.finished = True
                        print(f"[TOOL {i}] Task marked as finished!")
                except Exception as e:
                    result_text = f"Tool execution failed: {str(e)}"
                    print(f"[TOOL {i}] Error: {e}")
                tool_results.append(result_text)
                if state.finished:
                    break

            print("[WAIT] Waiting 2 seconds after tool execution...")
            await browser.page.wait_for_timeout(2000)

            # Check for open dropdowns
            print("[DROPDOWN] Checking for open dropdowns...")
            try:
                dropdown_options = await asyncio.wait_for(
                    self._check_dropdown_options(browser.page), timeout=5.0
                )
                if dropdown_options:
                    print(f"[DROPDOWN] Found options: {dropdown_options}")
                else:
                    print("[DROPDOWN] No open dropdowns found")
            except Exception as e:
                print(f"[DROPDOWN] Check failed: {e}")
                dropdown_options = None

            # Build result message with dropdown info if found
            result_parts = list(tool_results)
            if dropdown_options:
                dropdown_message = f"""

You just opened a system level dropdown, which cannot be shown in the screenshots so the options are displayed in text form below.
Use the select_dropdown tool to select an option from the dropdown. Use the exact value (the part in quotes) from the available dropdown options.

Available dropdown options: {dropdown_options}"""
                result_parts.append(dropdown_message)

            # Save result as simple user message
            state.messages.append({"role": "user", "content": "\n".join(result_parts)})
            print(f"[MESSAGES] Added tool results to conversation")

            # Log tool results to Opik trace
            try:
                trace.update(
                    output={
                        "tool_calls": [{"name": tc[0], "args": tc[1]} for tc in tool_calls],
                        "tool_results": tool_results,
                        "finished": state.finished
                    }
                )
                print(f"[OPIK] Updated trace with tool execution results")
            except Exception as e:
                print(f"[OPIK] Failed to update trace: {e}")
        else:
            # No tool calls - just log the response
            try:
                trace.update(
                    output={
                        "response": message.content,
                        "finished": state.finished
                    }
                )
                print(f"[OPIK] Updated trace with response")
            except Exception as e:
                print(f"[OPIK] Failed to update trace: {e}")

        # End trace
        try:
            trace.end()
            print(f"[OPIK] Trace ended successfully")
        except Exception as e:
            print(f"[OPIK] Failed to end trace: {e}")

        print(f"\n[STEP] Step complete. State finished: {state.finished}\n")
        return state
