import asyncio

from agi_agents.qwen.qwen import QwenAgent
from arena import RunHarness


async def main():

    agent = QwenAgent(
        api_key="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"
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
