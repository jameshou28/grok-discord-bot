import sys
import asyncio
from bot import RAGStore, logger

async def main(files: list[str]):
    for f in files:
        await RAGStore().load(f)

if __name__ == "__main__":
    files = sys.argv[1:] or ["knowledge.txt"]
    asyncio.run(main(files))
    logger.info("Embedding cache build complete.")
