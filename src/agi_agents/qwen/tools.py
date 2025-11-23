"""Tool execution for Qwen agent with text-based tool calling."""

from typing import Dict, Any, Tuple
import logging
from urllib.parse import urlparse

from patchright.async_api import Page


class QwenToolExecutor:
    """Execute tools for Qwen agent."""

    def __init__(
        self,
        page: Page,
        browser: "AgentBrowser",
        state: "AgentState",
        original_width: int,
        original_height: int,
    ):
        self.page = page
        self.browser = browser
        self.state = state
        self.original_width = original_width
        self.original_height = original_height
        self._logger = logging.getLogger(__name__)
        allowed_hosts = getattr(state, "allowed_hosts", set())
        self._allowed_hosts = {host.lower() for host in allowed_hosts}

    def scale_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Scale coordinates from 0-1000 range to actual viewport dimensions.

        Qwen returns coordinates in 0-1000 range, which we scale to actual image dimensions.
        This matches the behavior in qwen3vl.py.
        """
        scaled_x = int((x / 1000) * self.original_width)
        scaled_y = int((y / 1000) * self.original_height)
        return scaled_x, scaled_y

    def _summarize_input(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Return a log-friendly snapshot of tool arguments."""

        summarized: Dict[str, Any] = {}
        for key, value in tool_input.items():
            if key == "content" and isinstance(value, str):
                snippet = value.replace("\n", "\\n")
                summarized[key] = snippet[:80] + ("..." if len(snippet) > 80 else "")
            elif key.endswith("point_2d") and isinstance(value, (list, tuple)):
                summarized[key] = [round(v, 2) for v in value]
            elif key == "url" and isinstance(value, str):
                summarized[key] = value
            else:
                summarized[key] = value
        return summarized

    def _is_allowed_host(self, host: str) -> bool:
        host = host.lower()
        if not host:
            return True
        if host in self._allowed_hosts:
            return True
        for allowed in self._allowed_hosts:
            if allowed and host.endswith(f".{allowed}"):
                return True
        return False

    async def _get_expanded_select(self):
        """Return first <select> element-handle that is AX-expanded, else None."""
        try:
            selects = await self.page.query_selector_all("select")
            for sel in selects:
                try:
                    ax = await self.page.accessibility.snapshot(root=sel)
                    if ax and ax.get("expanded"):
                        return sel
                except Exception:
                    continue  # element went stale
        except Exception:
            pass  # navigation race
        return None

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return result message."""
        if self._logger.isEnabledFor(logging.INFO):
            summary = self._summarize_input(tool_name, tool_input)
            self._logger.info("Executing tool %s with args %s", tool_name, summary)
        if tool_name == "click":
            return await self._execute_click(tool_input)
        elif tool_name == "double_click":
            return await self._execute_double_click(tool_input)
        elif tool_name == "triple_click":
            return await self._execute_triple_click(tool_input)
        elif tool_name == "hover":
            return await self._execute_hover(tool_input)
        elif tool_name == "press_and_hold":
            return await self._execute_press_and_hold(tool_input)
        elif tool_name == "drag":
            return await self._execute_drag(tool_input)
        elif tool_name == "type":
            return await self._execute_type(tool_input)
        elif tool_name == "hotkey":
            return await self._execute_hotkey(tool_input)
        elif tool_name == "press":
            return await self._execute_press(tool_input)
        elif tool_name == "scroll":
            return await self._execute_scroll(tool_input)
        elif tool_name == "goto":
            return await self._execute_goto(tool_input)
        elif tool_name == "finished":
            return await self._execute_finished(tool_input)
        else:
            return f"Unknown tool: {tool_name}"

    async def _execute_click(self, tool_input: Dict[str, Any]) -> str:
        point = tool_input["point_2d"]
        x, y = self.scale_coordinates(point[0], point[1])
        await self.page.mouse.click(x, y)
        await self._maybe_reveal_reply_controls()
        return f"Clicked at ({point[0]}, {point[1]})"

    async def _execute_double_click(self, tool_input: Dict[str, Any]) -> str:
        point = tool_input["point_2d"]
        x, y = self.scale_coordinates(point[0], point[1])
        await self.page.mouse.dblclick(x, y)
        return f"Double clicked at ({point[0]}, {point[1]})"

    async def _execute_triple_click(self, tool_input: Dict[str, Any]) -> str:
        point = tool_input["point_2d"]
        x, y = self.scale_coordinates(point[0], point[1])
        await self.page.mouse.click(x, y, click_count=3)
        return f"Triple clicked at ({point[0]}, {point[1]})"

    async def _execute_hover(self, tool_input: Dict[str, Any]) -> str:
        point = tool_input["point_2d"]
        x, y = self.scale_coordinates(point[0], point[1])
        await self.page.mouse.move(x, y)
        return f"Hovered at ({point[0]}, {point[1]})"

    async def _execute_press_and_hold(self, tool_input: Dict[str, Any]) -> str:
        hold_ms = int(tool_input.get("milliseconds", 500))
        hold_ms = max(0, min(hold_ms, 2000))  # clamp to a sensible window

        if "key" in tool_input:
            key = tool_input["key"]
            await self.page.keyboard.down(key)
            try:
                if hold_ms:
                    await self.page.wait_for_timeout(hold_ms)
            finally:
                await self.page.keyboard.up(key)
            return f"Pressed and held key {key} for {hold_ms} ms"

        if "point_2d" not in tool_input:
            return "Press-and-hold skipped: no target coordinates provided."

        if not tool_input.get("allow_mouse_hold", False):
            return (
                "Press-and-hold skipped to keep focus stable. Use click or drag instead, or set allow_mouse_hold=true if a true hold is required."
            )

        point = tool_input["point_2d"]
        x, y = self.scale_coordinates(point[0], point[1])
        await self.page.mouse.move(x, y)

        await self.page.mouse.down()
        try:
            if hold_ms:
                await self.page.wait_for_timeout(hold_ms)
        finally:
            await self.page.mouse.up()

        return f"Pressed and held at ({point[0]}, {point[1]}) for {hold_ms} ms"

    async def _execute_drag(self, tool_input: Dict[str, Any]) -> str:
        start_point = tool_input["start_point_2d"]
        end_point = tool_input["end_point_2d"]
        start_x, start_y = self.scale_coordinates(start_point[0], start_point[1])
        end_x, end_y = self.scale_coordinates(end_point[0], end_point[1])

        await self.page.mouse.move(start_x, start_y)
        await self.page.mouse.down()
        await self.page.mouse.move(end_x, end_y)
        await self.page.mouse.up()
        return f"Dragged from ({start_point[0]}, {start_point[1]}) to ({end_point[0]}, {end_point[1]})"

    async def _execute_type(self, tool_input: Dict[str, Any]) -> str:
        content = tool_input["content"]
        await self.page.keyboard.type(content, delay=50)
        return f"Typed: {content[:50]}{'...' if len(content) > 50 else ''}"

    async def _execute_hotkey(self, tool_input: Dict[str, Any]) -> str:
        key = tool_input["key"]
        await self.page.keyboard.press(key)
        return f"Pressed hotkey: {key}"

    async def _execute_press(self, tool_input: Dict[str, Any]) -> str:
        key = tool_input["key"]
        await self.page.keyboard.press(key)
        return f"Pressed key: {key}"

    async def _execute_scroll(self, tool_input: Dict[str, Any]) -> str:
        direction = tool_input["direction"]
        pixels = tool_input.get("pixels", 300)

        # Establish the target point for focusing/scrolling
        scroll_x: float | None = None
        scroll_y: float | None = None

        if "point_2d" in tool_input:
            point = tool_input["point_2d"]
            x, y = self.scale_coordinates(point[0], point[1])
            await self.page.mouse.move(x, y)
            scroll_x, scroll_y = float(x), float(y)
        else:
            viewport_size = await self.page.evaluate(
                "({width: window.innerWidth, height: window.innerHeight})"
            )
            scroll_x = viewport_size["width"] / 2
            scroll_y = viewport_size["height"] / 2
            await self.page.mouse.move(scroll_x, scroll_y)

        # Focus the element under the cursor
        await self.page.wait_for_timeout(100)
        await self.page.evaluate(f"""
            const element = document.elementFromPoint({scroll_x}, {scroll_y});
            if (element && element.focus) {{
                element.focus();
            }}
        """)

        # Scroll in the specified direction
        delta_x = 0
        delta_y = 0
        magnitude = abs(pixels)

        if direction == "down":
            delta_y = magnitude
        elif direction == "up":
            delta_y = -magnitude
        elif direction == "right":
            delta_x = magnitude
        elif direction == "left":
            delta_x = -magnitude
        else:
            delta_y = pixels

        if delta_x == 0 and delta_y == 0:
            return "Scroll skipped: zero delta"

        scroll_result = None
        try:
            scroll_result = await self.page.evaluate(
                """
                ({ x, y, dx, dy }) => {
                    const detail = { scrolled: false, target: null };
                    const start = document.elementFromPoint(x, y);
                    const visited = new Set();

                    const couldScroll = (el, axis) => {
                        if (!el) return false;
                        const style = window.getComputedStyle(el);
                        if (axis === "y") {
                            if (el.scrollHeight <= el.clientHeight + 1) return false;
                            return ["auto", "scroll", "overlay"].includes(style.overflowY);
                        }
                        if (axis === "x") {
                            if (el.scrollWidth <= el.clientWidth + 1) return false;
                            return ["auto", "scroll", "overlay"].includes(style.overflowX);
                        }
                        return false;
                    };

                    let node = start;
                    while (node) {
                        if (visited.has(node)) break;
                        visited.add(node);

                        const canScrollY = Math.abs(dy) > 0 && couldScroll(node, "y");
                        const canScrollX = Math.abs(dx) > 0 && couldScroll(node, "x");
                        if (canScrollY || canScrollX) {
                            node.scrollBy({ left: dx, top: dy, behavior: "instant" });
                            detail.scrolled = true;
                            detail.target = node.tagName || "UNKNOWN";
                            return detail;
                        }
                        node = node.parentElement;
                    }

                    window.scrollBy({ left: dx, top: dy, behavior: "instant" });
                    detail.scrolled = true;
                    detail.target = "WINDOW";
                    return detail;
                }
                """,
                {
                    "x": scroll_x,
                    "y": scroll_y,
                    "dx": delta_x,
                    "dy": delta_y,
                },
            )
        except Exception as exc:  # guard against DOM exceptions
            self._logger.debug("Scroll JS fallback due to: %s", exc)

        if not scroll_result or not scroll_result.get("scrolled"):
            # fall back to the native mouse wheel if DOM manipulation failed
            await self.page.mouse.wheel(delta_x, delta_y)
            target = "wheel"
        else:
            target = scroll_result.get("target", "unknown")

        return f"Scrolled {direction} by {pixels} pixels (target={target})"

    async def _execute_goto(self, tool_input: Dict[str, Any]) -> str:
        url = tool_input["url"].strip()
        parsed = urlparse(url)
        if not self._is_allowed_host(parsed.netloc):
            self._logger.warning(
                "Blocked navigation to disallowed host %s (allowed=%s)",
                parsed.netloc,
                ", ".join(sorted(self._allowed_hosts)) or "<none>",
            )
            return (
                "Navigation blocked: stay within the task environment and use on-page "
                "controls instead of visiting external sites."
            )

        self._logger.info("Navigating via goto to %s", url)
        await self.page.goto(url)
        if parsed.netloc:
            normalized = parsed.netloc.lower()
            self._allowed_hosts.add(normalized)
            if hasattr(self.state, "allowed_hosts"):
                self.state.allowed_hosts.add(normalized)
        return f"Navigated to: {url}"

    async def _execute_finished(self, tool_input: Dict[str, Any]) -> str:
        content = tool_input["content"]
        return f"Task finished: {content}"

    async def _maybe_reveal_reply_controls(self) -> None:
        """Scroll gomail threads to expose reply controls after opening messages."""

        try:
            host = urlparse(self.page.url).netloc.lower()
        except Exception:
            return

        if host != "real-gomail.vercel.app":
            return

        try:
            await self.page.evaluate(
                """
                (() => {
                    const buttons = Array.from(document.querySelectorAll('button'));
                    const target = buttons.find(btn => {
                        const label = (btn.textContent || btn.getAttribute('aria-label') || '').trim().toLowerCase();
                        return label === 'reply' || label === 'reply all' || label === 'send reply';
                    });
                    if (!target) return;

                    const rect = target.getBoundingClientRect();
                    const fullyVisible = rect.top >= 0 && rect.bottom <= window.innerHeight;
                    if (!fullyVisible) {
                        target.scrollIntoView({ block: 'center', behavior: 'instant' });
                    }
                })();
                """
            )
        except Exception:
            # Ignore DOM churn; the agent will fall back to manual scroll if needed.
            return
