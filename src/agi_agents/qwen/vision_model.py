"""
QwenVisionModel for page analysis and element location.

This module implements the vision model component of the adaptive agent system,
using Qwen VLM to analyze screenshots and locate elements on web pages.
"""

import base64
import io
import json
import time
from typing import List, Dict, Any, Union, Optional

from openai import AsyncOpenAI
from PIL import Image

import opik
from opik import Attachment

from agi_agents.models import (
    PageElement,
    PageAnalysis,
    ElementLocation,
    ElementLocationFailure,
)


class QwenVisionModel:
    """
    Qwen VLM for visual page analysis and element location.
    Analyzes screenshots in parallel with GPT-4o's goal analysis.
    
    This model is responsible for:
    1. Analyzing screenshots to identify ALL interactive elements with coordinates
    2. Responding to straightforward element location requests from the orchestrator
    3. Reporting failures when elements cannot be located
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-vl-235b-a22b-thinking",
        client: Optional[AsyncOpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Qwen Vision Model.
        
        Args:
            model: Model identifier for Qwen VLM
            client: Optional pre-configured AsyncOpenAI client
            api_key: API key for OpenAI-compatible endpoint
            base_url: Base URL for OpenAI-compatible endpoint
        """
        self.model = model
        
        if client is not None:
            self.client = client
        else:
            # Use environment variable if available, otherwise use provided key
            api_key = api_key or "your-openrouter-api-key"
            base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        
        # Initialize Opik client for tracing
        self.opik_client = opik.Opik(project_name="agi")
        print(f"[OPIK] Initialized Opik client for vision model with project: agi")
        
        # Min/max pixels for Qwen vision model
        self.min_pixels = 64 * 32 * 32
        self.max_pixels = 9800 * 32 * 32

    def _screenshot_to_data_url(self, screenshot: bytes) -> str:
        """Convert screenshot bytes to data URL for API."""
        b64 = base64.b64encode(screenshot).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    async def analyze_page(
        self,
        screenshot: bytes,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> PageAnalysis:
        """
        Analyze screenshot and identify ALL interactive elements.
        This runs in parallel with GPT-4o's goal analysis.
        
        Args:
            screenshot: Screenshot bytes (JPEG format)
            history: Optional conversation history for context
            
        Returns:
            PageAnalysis with ALL elements, types, labels, and coordinates
            
        Raises:
            RuntimeError: If API call fails after retries
        """
        print(f"[VISION_MODEL] analyze_page called, screenshot size: {len(screenshot)} bytes")
        print(f"[VISION_MODEL] Converting screenshot to data URL...")
        data_url = self._screenshot_to_data_url(screenshot)
        print(f"[VISION_MODEL] Data URL created, building prompt...")
        
        # Build prompt for comprehensive page analysis
        analysis_prompt = """Analyze this screenshot and identify ALL interactive elements on the page.

For each interactive element, provide:
1. A unique element_id (e.g., "btn_1", "input_1", "link_1")
2. The element_type (button, input, select, link, checkbox, radio, textarea, etc.)
3. The visible label or text
4. The center coordinates as [x, y] in pixels
5. For input elements, the field_type (text, email, password, number, date, tel, etc.)

Also provide:
- page_type: Classification of the page (form, search, content, navigation, etc.)
- content_summary: Brief summary of what's on the page

Return your analysis as a JSON object with this structure:
{
  "elements": [
    {
      "element_id": "btn_1",
      "element_type": "button",
      "label": "Submit",
      "coordinates": [450, 300],
      "field_type": null,
      "attributes": {}
    }
  ],
  "page_type": "form",
  "content_summary": "Login form with username and password fields"
}

Be thorough - identify ALL interactive elements you can see."""

        print(f"[VISION_MODEL] Building messages for API call...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": analysis_prompt},
                ],
            }
        ]
        print(f"[VISION_MODEL] Messages built successfully")

        # Create Opik trace
        print("[OPIK] Creating trace for vision model call...")
        trace = self.opik_client.trace(
            name=f"qwen_vision_analyze_page",
            input={
                "model": self.model,
                "screenshot_size": len(screenshot),
                "prompt": analysis_prompt[:500],  # First 500 chars of prompt
                "has_image": True
            },
            metadata={"model": self.model, "screenshot_size": len(screenshot)}
        )
        
        # Save screenshot to a temporary file for Opik attachment
        screenshot_path = None
        try:
            import tempfile
            screenshot_image = Image.open(io.BytesIO(screenshot))
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as tmp_file:
                screenshot_path = tmp_file.name
                screenshot_image.save(tmp_file, format='JPEG', quality=80)
            print(f"[OPIK] Saved screenshot to temp file: {screenshot_path}")
        except Exception as e:
            print(f"[OPIK] Failed to save screenshot: {e}")

        # Call API with retry logic and exponential backoff
        max_retries = 3
        last_error = None
        
        print(f"[VISION_MODEL] Starting page analysis with model {self.model}...")
        for attempt in range(max_retries):
            try:
                # Add exponential backoff delay before retry (except first attempt)
                if attempt > 0:
                    import asyncio
                    backoff_delay = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                    await asyncio.sleep(backoff_delay)
                
                print(f"[VISION_MODEL] Calling API (attempt {attempt + 1}/{max_retries})...")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=2048,
                    messages=messages,
                )
                print(f"[VISION_MODEL] API call completed successfully")

                # Check for API errors
                if hasattr(response, "error") and response.error:
                    error_msg = response.error.get("message", "Unknown error")
                    last_error = f"API error: {error_msg}"
                    print(f"[VISION_MODEL] API error detected: {last_error}")
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError(last_error)

                # Check for valid response
                if not response.choices or len(response.choices) == 0:
                    last_error = "Empty response from API"
                    print(f"[VISION_MODEL] Error: {last_error}")
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError(last_error)

                content = response.choices[0].message.content
                if not content:
                    last_error = "Empty content in response"
                    print(f"[VISION_MODEL] Error: {last_error}")
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError(last_error)
                
                print(f"[VISION_MODEL] Received response content (length: {len(content)})")
                print(f"[VISION_MODEL] Content preview: {content[:200]}...")

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

                # Build PageAnalysis from response
                elements = []
                for elem_data in data.get("elements", []):
                    element = PageElement(
                        element_id=elem_data["element_id"],
                        element_type=elem_data["element_type"],
                        label=elem_data["label"],
                        coordinates=tuple(elem_data["coordinates"]),
                        field_type=elem_data.get("field_type"),
                        attributes=elem_data.get("attributes", {}),
                    )
                    elements.append(element)

                page_analysis = PageAnalysis(
                    elements=elements,
                    page_type=data.get("page_type", "unknown"),
                    content_summary=data.get("content_summary", ""),
                    timestamp=time.time(),
                )
                
                # Log to Opik
                try:
                    attachments = []
                    if screenshot_path:
                        attachments.append(
                            Attachment(
                                data=screenshot_path,
                                content_type="image/jpeg",
                                name="screenshot_vision_analysis.jpg"
                            )
                        )
                    
                    trace.span(
                        name="vision_model_analyze",
                        type="llm",
                        input={
                            "messages": str(messages)[:1000],  # Truncate for readability
                            "model": self.model,
                            "prompt_preview": analysis_prompt[:200]
                        },
                        output={
                            "elements_count": len(page_analysis.elements),
                            "page_type": page_analysis.page_type,
                            "content_summary": page_analysis.content_summary
                        },
                        model=self.model,
                        metadata={
                            "attempt": attempt + 1,
                            "temperature": 0.0,
                            "max_tokens": 2048
                        },
                        attachments=attachments
                    )
                    print(f"[OPIK] Logged vision model call to trace")
                except Exception as e:
                    print(f"[OPIK] Failed to log vision model call: {e}")
                
                # Update trace with output
                try:
                    trace.update(
                        output={
                            "page_analysis": {
                                "elements_count": len(page_analysis.elements),
                                "page_type": page_analysis.page_type,
                                "content_summary": page_analysis.content_summary
                            }
                        }
                    )
                    trace.end()
                    print(f"[OPIK] Trace ended successfully")
                except Exception as e:
                    print(f"[OPIK] Failed to end trace: {e}")
                
                # Fallback logic for empty PageAnalysis
                if not page_analysis.elements:
                    # Return a minimal fallback analysis with basic page info
                    return PageAnalysis(
                        elements=[],
                        page_type="unknown",
                        content_summary="Unable to identify interactive elements on this page",
                        timestamp=time.time(),
                    )
                
                return page_analysis

            except json.JSONDecodeError as e:
                last_error = f"Failed to parse JSON response: {e}"
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(last_error) from e
            except Exception as e:
                last_error = f"API call failed: {e}"
                if attempt < max_retries - 1:
                    continue
                raise RuntimeError(last_error) from e

        # Fallback: Return empty PageAnalysis instead of raising
        # This allows the system to continue with a minimal analysis
        try:
            trace.update(output={"error": last_error, "fallback": True})
            trace.end()
            print(f"[OPIK] Trace ended with fallback")
        except Exception as e:
            print(f"[OPIK] Failed to end trace: {e}")
        
        return PageAnalysis(
            elements=[],
            page_type="unknown",
            content_summary=f"Vision model failed after {max_retries} retries: {last_error}",
            timestamp=time.time(),
        )

    async def locate_element(
        self,
        screenshot: bytes,
        element_description: str,
        page_analysis: PageAnalysis,
    ) -> Union[ElementLocation, ElementLocationFailure]:
        """
        Respond to straightforward element location request from GPT-4o.
        Called ONLY when GPT-4o has a concrete plan to use the element.
        
        Args:
            screenshot: Current screenshot bytes
            element_description: Straightforward request (e.g., "find the create button")
            page_analysis: Current page analysis with available elements
            
        Returns:
            ElementLocation with coordinates, or ElementLocationFailure if not found
            
        Raises:
            RuntimeError: If API call fails after retries
        """
        data_url = self._screenshot_to_data_url(screenshot)
        
        # Build list of available elements for context
        available_elements = [
            f"{elem.element_id}: {elem.element_type} - '{elem.label}'"
            for elem in page_analysis.elements
        ]
        
        # Build prompt for element location
        location_prompt = f"""Find the specific element: {element_description}

Available elements on the page:
{chr(10).join(available_elements)}

If you can locate the requested element, respond with JSON:
{{
  "found": true,
  "element_id": "the_element_id",
  "coordinates": [x, y],
  "element_type": "button",
  "confidence": 0.95
}}

If you CANNOT locate the element, respond with JSON:
{{
  "found": false,
  "reason": "explanation of why not found",
  "available_elements": ["list", "of", "available", "element", "labels"]
}}

Be precise - only return found=true if you're confident you found the right element."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": location_prompt},
                ],
            }
        ]

        # Call API with retry logic and exponential backoff
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Add exponential backoff delay before retry (except first attempt)
                if attempt > 0:
                    import asyncio
                    backoff_delay = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                    await asyncio.sleep(backoff_delay)
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=1024,
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

                # Check if element was found
                if data.get("found", False):
                    return ElementLocation(
                        element_id=data["element_id"],
                        coordinates=tuple(data["coordinates"]),
                        element_type=data["element_type"],
                        confidence=data.get("confidence", 1.0),
                    )
                else:
                    # Element not found - return failure
                    return ElementLocationFailure(
                        element_description=element_description,
                        reason=data.get("reason", "Element not found"),
                        available_elements=data.get(
                            "available_elements",
                            [elem.label for elem in page_analysis.elements],
                        ),
                    )

            except json.JSONDecodeError as e:
                last_error = f"Failed to parse JSON response: {e}"
                if attempt < max_retries - 1:
                    continue
                # On final attempt, return failure instead of raising
                return ElementLocationFailure(
                    element_description=element_description,
                    reason=f"Failed to parse vision model response: {e}",
                    available_elements=[elem.label for elem in page_analysis.elements],
                )
            except Exception as e:
                last_error = f"API call failed: {e}"
                if attempt < max_retries - 1:
                    continue
                # On final attempt, return failure instead of raising
                return ElementLocationFailure(
                    element_description=element_description,
                    reason=f"Vision model API error: {e}",
                    available_elements=[elem.label for elem in page_analysis.elements],
                )

        # Should not reach here, but return failure as fallback
        return ElementLocationFailure(
            element_description=element_description,
            reason=f"locate_element failed after {max_retries} retries: {last_error}",
            available_elements=[elem.label for elem in page_analysis.elements],
        )
