import asyncio
import logging

from agi_agents.qwen.qwen import QwenAgent
from arena import RunHarness
from arena.logging_config import configure_logging


async def main():
    configure_logging(level="DEBUG")
    logging.debug("Logging configured for bench script")
    agent = QwenAgent()

    harness = RunHarness(
        agent=agent,
        tasks=[
            "src/benchmarks/hackathon/tasks/gomail-5.json"
        ],
        parallel=1,
        sample_count=1,
        max_steps=60,
        headless=False,
    )

    results = await harness.run()
    logging.debug("Run completed with results: %s", results)


if __name__ == "__main__":
    asyncio.run(main())
