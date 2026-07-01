import discord
from google import genai
from google.genai import types
import os
import numpy as np
import asyncio
import logging
from dotenv import load_dotenv
from typing import List
import hashlib
import pickle

# --- LOGGING CONFIGURATION ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# Create handlers: one for the file, one for the terminal
file_handler = logging.FileHandler('bot_logs.txt', encoding='utf-8')
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=date_format,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger('grok_bot')
# -----------------------------

load_dotenv()

# API clients
keys_env = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]
current_key_idx = 0

if not API_KEYS:
    logger.error("No Gemini API keys found in environment variables!")

CLIENTS = [genai.Client(api_key=k) for k in API_KEYS]

def current_client() -> genai.Client:
    return CLIENTS[current_key_idx]

# RAG for general knowledge
def chunk_text(text: str, chunk_size: int = 120) -> List[str]:
    words = text.split()
    step = chunk_size // 2
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step) if words[i : i + chunk_size]]

async def embed_texts(texts: list[str]) -> list:
    global current_key_idx
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        keys_tried = 0
        max_keys = len(CLIENTS)

        while keys_tried < max_keys:
            try:
                result = await CLIENTS[current_key_idx].aio.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch
                )
                all_embeddings.extend([np.array(e.values) for e in result.embeddings])  # convert here
                break
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "429" in error_str:
                    keys_tried += 1
                    old_idx = current_key_idx
                    current_key_idx = (current_key_idx + 1) % max_keys
                    logger.warning(f"Embed: Key {old_idx} hit quota. Switching to key {current_key_idx}...")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Embed error: {e}")
                    raise
        else:
            logger.error("Embed: all keys exhausted on this batch.")
            raise RuntimeError("All API keys quota-exhausted during embedding.")

    return all_embeddings

async def embed_query(query: str) -> np.ndarray:
    if not CLIENTS: return np.zeros(768)
    result = await current_client().aio.models.embed_content(model="gemini-embedding-001", contents=[query])
    return np.array(result.embeddings[0].values)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0

class RAGStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.ready = False

    def _cache_path(self, filepath: str) -> str:
        # One cache file per source file, stored next to it
        return filepath + ".embed_cache.pkl"

    def _file_hash(self, filepath: str) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    async def load(self, filepath: str):
        if not os.path.exists(filepath):
            logger.error(f"Knowledge file '{filepath}' not found.")
            return

        cache_path = self._cache_path(filepath)
        current_hash = self._file_hash(filepath)

        # Try loading from cache
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("hash") == current_hash:
                self.chunks += cached["chunks"]
                self.embeddings += cached["embeddings"]
                self.ready = True
                logger.info(f"Loaded {len(cached['chunks'])} chunks from cache for '{filepath}'.")
                return
            else:
                logger.info(f"Cache stale for '{filepath}', re-embedding...")

        # No valid cache — embed and save
        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        new_chunks = chunk_text(text)
        new_embeddings = await embed_texts(new_chunks)

        with open(cache_path, "wb") as f:
            pickle.dump({
                "hash": current_hash,
                "chunks": new_chunks,
                "embeddings": new_embeddings,
            }, f)

        self.chunks += new_chunks
        self.embeddings += new_embeddings
        self.ready = True
        logger.info(f"Embedded and cached {len(new_chunks)} chunks for '{filepath}'.")
    async def retrieve(self, query: str, top_k: int = 5) -> str:
        if not self.ready or not self.chunks:
            return "Knowledge base not loaded."
        
        query_vec = await embed_query(query)
        scores = [cosine_similarity(query_vec, e) for e in self.embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_chunks = [self.chunks[i] for i in top_indices]
        
        logger.info(f"RAG retrieved {len(top_chunks)} chunks for query: '{query}'")
        return "\n\n".join(top_chunks)

rag_store = RAGStore()

async def search_knowledge_fn(query: str) -> str:
    logger.info(f"Tool Call: search_knowledge for query: '{query}'")
    return await rag_store.retrieve(query)

# Tools
TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_knowledge",
            description="Search the knowledge base.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"query": types.Schema(type=types.Type.STRING)},
                required=["query"],
            ),
        ),
    ]
)

TOOL_MAP = {
    "search_knowledge": search_knowledge_fn,
}

DEFAULT_INSTRUCTION = (
    "You are a helpful assistant in a Discord server. Keep replies concise, plain text only.\n"
    "If the user asks something that may be covered by the knowledge base, use search_knowledge "
    "and give a straightforward answer based on what you find."
)

async def generate(prompt: str, history: List[types.Content], system_instruction: str = DEFAULT_INSTRUCTION) -> str:
    global current_key_idx
    history = list(history)
    history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    config = types.GenerateContentConfig(system_instruction=system_instruction, tools=[TOOLS])

    logic_turns = 0
    while logic_turns < 5:
        keys_tried_this_turn = 0
        max_keys = len(CLIENTS)
        
        response = None
        while keys_tried_this_turn < max_keys:
            try:
                logger.info(f"Gemini Request using Key Index {current_key_idx} (Logic Turn {logic_turns + 1})")
                response = await current_client().aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=history,
                    config=config,
                )
                break 
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "429" in error_str:
                    keys_tried_this_turn += 1
                    old_idx = current_key_idx
                    current_key_idx = (current_key_idx + 1) % max_keys
                    logger.warning(f"Key {old_idx} hit quota. Trying next key ({current_key_idx})...")
                    await asyncio.sleep(1) 
                    continue
                else:
                    logger.error(f"Gemini API error: {e}")
                    raise e

        if response is None:
            logger.error("ALL API KEYS EXHAUSTED.")
            return "all my brain cells are fried. try again later."

        logic_turns += 1
        parts = response.candidates[0].content.parts
        function_calls = [p.function_call for p in parts if p.function_call]
        history.append(types.Content(role="model", parts=parts))

        if not function_calls:
            final_text = "".join(p.text for p in parts if p.text)
            logger.info(f"Model Response: {final_text}")
            return final_text

        tool_result_parts = []
        for fc in function_calls:
            fn = TOOL_MAP.get(fc.name)
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**fc.args)
            else:
                result = fn(**fc.args)
            tool_result_parts.append(types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": result})))
        history.append(types.Content(role="user", parts=tool_result_parts))
    
    return "Error: too many tool calls."

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    logger.info(f"Logged in to Discord as {client.user}")
    await rag_store.load("knowledge.txt")

@client.event
async def on_message(message: discord.Message):
    if message.author == client.user or not client.user.mentioned_in(message): return
    prompt = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not prompt: return
    
    logger.info(f"Incoming Mention from {message.author.display_name}: '{prompt}'")

    full_prompt = f"[User: {message.author.display_name}] {prompt}"

    async with message.channel.typing():
        try:
            msg = await generate(full_prompt, history=[])
            await message.reply(msg[:2000])
        except Exception as e:
            logger.error(f"Failed to handle message: {e}", exc_info=True)
            await message.reply(f"error: {e}")

client.run(os.getenv("DISCORD_TOKEN"))