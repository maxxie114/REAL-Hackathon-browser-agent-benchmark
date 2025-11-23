"""
Test script to verify AdaptiveAgent with Opik logging for orchestrator.
"""
import asyncio
from openai import AsyncOpenAI
from agi_agents.adaptive_agent import AdaptiveAgent
from agi_agents.qwen.vision_model import QwenVisionModel
from agi_agents.orchestrator import GPT4Orchestrator
from agi_agents.metrics_tracker import MetricsTracker
from arena import RunHarness


async def main():
    # Initialize components
    api_key = "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"
    base_url = "https://openrouter.ai/api/v1"
    
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    vision_model = QwenVisionModel(
        model="qwen/qwen3-vl-235b-a22b-thinking",
        client=client
    )
    
    orchestrator_model = GPT4Orchestrator(
        model="openai/gpt-4o",
        client=client
    )
    
    metrics_tracker = MetricsTracker()
    
    # Create AdaptiveAgent instance
    agent = AdaptiveAgent(
        vision_model=vision_model,
        orchestrator_model=orchestrator_model,
        metrics_tracker=metrics_tracker
    )
    
    print("[DEBUG] Creating RunHarness with AdaptiveAgent...")
    harness = RunHarness(
        agent=agent,
        tasks=[
            "src/benchmarks/hackathon/tasks/gomail-6.json"
        ],
        parallel=1,
        sample_count=1,
        max_steps=60,
        headless=True,
    )
    results = await harness.run()
    print(f"\n[RESULTS] {results}")


if __name__ == "__main__":
    asyncio.run(main())
