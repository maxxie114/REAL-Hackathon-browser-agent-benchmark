import asyncio

from agi_agents.qwen.qwen import QwenAgent
from arena import RunHarness


async def main():

    agent = QwenAgent(
        api_key="your-openrouter-api-key"
    )

    print("[DEBUG] Creating RunHarness...")
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


if __name__ == "__main__":
    asyncio.run(main())
