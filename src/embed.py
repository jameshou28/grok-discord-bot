import sys
import os
import asyncio
from bot import RAGStore, logger, DATA_DIR

async def main(files: list[str]):
    for f in files:
        await RAGStore().load(f)

if __name__ == "__main__":
    files = sys.argv[1:] or [os.path.join(DATA_DIR, "knowledge.txt")]
    asyncio.run(main(files))
    logger.info("Embedding cache build complete.")
