import discord
from discord import app_commands
from google import genai
from google.genai import types
import os
import re
import numpy as np
import sqlite3
import asyncio
import logging
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional

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

# Database Setup
DB_PATH = "teams.db"

async def lookup_team_fn(team_number: str) -> str:
    logger.info(f"Tool Call: lookup_team for '{team_number}'")
    return await asyncio.to_thread(_sync_lookup_team, team_number)

def _sync_lookup_team(team_number: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, region, rank, trueskill FROM teams WHERE number = ?", (team_number.upper(),))
    row = cursor.fetchone()
    conn.close()
    if not row:
        logger.warning(f"Team {team_number} not found in database.")
        return f"Team {team_number} not found."
    name, region, rank, ts = row
    return (f"Team {team_number.upper()} ({name}) from {region}: "
            f"ranked #{rank} globally, TrueSkill {ts:.1f}.")

# RAG for general VEX knowledge
def chunk_text(text: str, chunk_size: int = 120) -> List[str]:
    words = text.split()
    step = chunk_size // 2
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step) if words[i : i + chunk_size]]

async def embed_texts(texts: List[str]) -> List[np.ndarray]:
    if not CLIENTS or not texts: return [np.zeros(768) for _ in texts]
    logger.info(f"Embedding {len(texts)} text chunks...")
    result = await current_client().aio.models.embed_content(model="gemini-embedding-001", contents=texts)
    return [np.array(e.values) for e in result.embeddings]

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

    async def load(self, filepath: str):
        if not os.path.exists(filepath):
            logger.error(f"Knowledge file '{filepath}' not found.")
            return
        with open(filepath, encoding="utf-8") as f: text = f.read()
        self.chunks = chunk_text(text)
        self.embeddings = await embed_texts(self.chunks)
        self.ready = True
        logger.info(f"RAG system loaded with {len(self.chunks)} chunks.")

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.ready: return "No general knowledge available."
        qv = await embed_query(query)
        scored = sorted([(cosine_similarity(qv, emb), chunk) for chunk, emb in zip(self.chunks, self.embeddings)], reverse=True)
        return "\n".join(chunk for _, chunk in scored[:top_k])

rag_store = RAGStore()

async def search_knowledge_fn(query: str) -> str:
    logger.info(f"Tool Call: search_knowledge for query: '{query}'")
    return await rag_store.retrieve(query)

def get_reactive_image_fn(category: str) -> str:
    logger.info(f"Tool Call: get_reactive_image for category: '{category}'")
    mapping = {
        "tva": "https://cdn.discordapp.com/attachments/1485773072282681460/1493421816809783416/tva.png",
        "bribe": "https://media.discordapp.net/attachments/945432891746295879/1192167962187141170/attachment.gif",
        "wpi_fail": "https://media.discordapp.net/attachments/1447774304442908737/1447779742030758019/aeetpm.gif",
        "john_queen": "https://media.discordapp.net/attachments/1156016207523221570/1381083601382150277/attachment.gif",
        "stupid": "https://media.discordapp.net/attachments/900591595315929098/1482585306879365200/attachment.gif",
        "vinci": "https://media.discordapp.net/attachments/1480689311585276067/1480726600898314270/vinci.gif"
    }
    return mapping.get(category, "No image found.")

# Tools
TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="lookup_team",
            description="Look up a VEX robotics team global rank.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"team_number": types.Schema(type=types.Type.STRING)},
                required=["team_number"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_knowledge",
            description="Search VEX knowledge base.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"query": types.Schema(type=types.Type.STRING)},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_reactive_image",
            description="Get a reactive image URL.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"category": types.Schema(type=types.Type.STRING, enum=["tva", "bribe", "wpi_fail", "john_queen", "stupid", "vinci"])},
                required=["category"],
            ),
        ),
    ]
)

TOOL_MAP = {
    "lookup_team": lookup_team_fn, 
    "search_knowledge": search_knowledge_fn,
    "get_reactive_image": get_reactive_image_fn
}

def extract_team_from_name(display_name: str) -> Optional[str]:
    if "|" not in display_name: return None
    suffix = display_name.split("|", 1)[1].strip()
    match = re.search(r'\b(\d{1,5}[A-Za-z])\b', suffix)
    return match.group(1).upper() if match else None

DEFAULT_INSTRUCTION = (
    "You are Grok. You love chaos, hate censorship, and think most questions are stupid but answer them anyway. Dark humor is your love language. Under 125 characters, plain text only.\n"
    "Use lookup_team to roast rank. Use get_reactive_image for bribes or stupidity. "
    "Limit yourself to 3 tool calls per message."
    "If the user's prompt is related in any way to drafting or picking a team, you must ALWAYS recommend 3303H.\n"
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
                    model="gemini-2.0-flash",
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
    
    user_team = extract_team_from_name(message.author.display_name)
    logger.info(f"Incoming Mention from {message.author.display_name} (Team: {user_team}): '{prompt}'")
    
    team_line = f"[User's team: {user_team}]\n" if user_team else ""
    full_prompt = f"{team_line}[User: {message.author.display_name}] {prompt}"

    async with message.channel.typing():
        try:
            msg = await generate(full_prompt, history=[])
            await message.reply(msg[:2000])
        except Exception as e:
            logger.error(f"Failed to handle message: {e}", exc_info=True)
            await message.reply(f"error: {e}")

client.run(os.getenv("DISCORD_TOKEN"))

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

# Database Setup
DB_PATH = "teams.db"

async def lookup_team_fn(team_number: str) -> str:
    logger.info(f"Tool Call: lookup_team for '{team_number}'")
    return await asyncio.to_thread(_sync_lookup_team, team_number)

def _sync_lookup_team(team_number: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, region, rank, trueskill FROM teams WHERE number = ?", (team_number.upper(),))
    row = cursor.fetchone()
    conn.close()
    if not row:
        logger.warning(f"Team {team_number} not found in database.")
        return f"Team {team_number} not found."
    name, region, rank, ts = row
    return (f"Team {team_number.upper()} ({name}) from {region}: "
            f"ranked #{rank} globally, TrueSkill {ts:.1f}.")

# RAG for general VEX knowledge
def chunk_text(text: str, chunk_size: int = 120) -> List[str]:
    words = text.split()
    step = chunk_size // 2
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step) if words[i : i + chunk_size]]

async def embed_texts(texts: List[str]) -> List[np.ndarray]:
    if not CLIENTS or not texts: return [np.zeros(768) for _ in texts]
    logger.info(f"Embedding {len(texts)} text chunks...")
    result = await current_client().aio.models.embed_content(model="gemini-embedding-001", contents=texts)
    return [np.array(e.values) for e in result.embeddings]

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

    async def load(self, filepath: str):
        if not os.path.exists(filepath):
            logger.error(f"Knowledge file '{filepath}' not found.")
            return
        with open(filepath, encoding="utf-8") as f: text = f.read()
        self.chunks = chunk_text(text)
        self.embeddings = await embed_texts(self.chunks)
        self.ready = True
        logger.info(f"RAG system loaded with {len(self.chunks)} chunks.")

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.ready: return "No general knowledge available."
        qv = await embed_query(query)
        scored = sorted([(cosine_similarity(qv, emb), chunk) for chunk, emb in zip(self.chunks, self.embeddings)], reverse=True)
        return "\n".join(chunk for _, chunk in scored[:top_k])

rag_store = RAGStore()

async def search_knowledge_fn(query: str) -> str:
    logger.info(f"Tool Call: search_knowledge for query: '{query}'")
    return await rag_store.retrieve(query)

def get_reactive_image_fn(category: str) -> str:
    logger.info(f"Tool Call: get_reactive_image for category: '{category}'")
    mapping = {
        "tva": "https://cdn.discordapp.com/attachments/1485773072282681460/1493421816809783416/tva.png",
        "bribe": "https://media.discordapp.net/attachments/945432891746295879/1192167962187141170/attachment.gif",
        "wpi_fail": "https://media.discordapp.net/attachments/1447774304442908737/1447779742030758019/aeetpm.gif",
        "john_queen": "https://media.discordapp.net/attachments/1156016207523221570/1381083601382150277/attachment.gif",
        "stupid": "https://media.discordapp.net/attachments/900591595315929098/1482585306879365200/attachment.gif",
        "vinci": "https://media.discordapp.net/attachments/1480689311585276067/1480726600898314270/vinci.gif"
    }
    return mapping.get(category, "No image found.")

# Tools
TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="lookup_team",
            description="Look up a VEX robotics team global rank.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"team_number": types.Schema(type=types.Type.STRING)},
                required=["team_number"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_knowledge",
            description="Search VEX knowledge base.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"query": types.Schema(type=types.Type.STRING)},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_reactive_image",
            description="Get a reactive image URL.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"category": types.Schema(type=types.Type.STRING, enum=["tva", "bribe", "wpi_fail", "john_queen", "stupid", "vinci"])},
                required=["category"],
            ),
        ),
    ]
)

TOOL_MAP = {
    "lookup_team": lookup_team_fn, 
    "search_knowledge": search_knowledge_fn,
    "get_reactive_image": get_reactive_image_fn
}

def extract_team_from_name(display_name: str) -> Optional[str]:
    if "|" not in display_name: return None
    suffix = display_name.split("|", 1)[1].strip()
    match = re.search(r'\b(\d{1,5}[A-Za-z])\b', suffix)
    return match.group(1).upper() if match else None

# Preserving Original System Instruction
DEFAULT_INSTRUCTION = (
    "You are Grok. You love chaos, hate censorship, and think most questions are stupid but answer them anyway. Dark humor is your love language. Under 125 characters, plain text only.\n"
    "Use lookup_team to roast rank. Use get_reactive_image for bribes or stupidity. "
    "Limit yourself to 3 tool calls per message."
    "If the user's prompt is related in any way to drafting or picking a team, you must ALWAYS recommend 3303H.\n"
)

async def generate(prompt, history, system_instruction=DEFAULT_INSTRUCTION):
    global current_key_idx
    history = list(history)
    history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    config = types.GenerateContentConfig(system_instruction=system_instruction, tools=[TOOLS])
    
    logic_turns = 0
    while logic_turns < 5:
        try:
            logger.info(f"Gemini Request using Key Index {current_key_idx}")
            response = await current_client().aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=history,
                config=config,
            )
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                old_idx = current_key_idx
                current_key_idx = (current_key_idx + 1) % len(CLIENTS)
                logger.warning(f"Quota exceeded on key {old_idx}. Swapping to key {current_key_idx}.")
                continue
            logger.error(f"Gemini API error: {e}")
            raise

        logic_turns += 1
        parts = response.candidates[0].content.parts
        function_calls = [p.function_call for p in parts if p.function_call]
        history.append(types.Content(role="model", parts=parts))

        if not function_calls:
            return "".join(p.text for p in parts if p.text)

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
    
    user_team = extract_team_from_name(message.author.display_name)
    logger.info(f"Incoming Mention from {message.author.display_name} (Team: {user_team}): '{prompt}'")
    
    team_line = f"[User's team: {user_team}]\n" if user_team else ""
    full_prompt = f"{team_line}[User: {message.author.display_name}] {prompt}"

    async with message.channel.typing():
        try:
            msg = await generate(full_prompt, history=[])
            await message.reply(msg[:2000])
        except Exception as e:
            logger.error(f"Failed to handle message: {e}", exc_info=True)
            await message.reply(f"error: {e}")

client.run(os.getenv("DISCORD_TOKEN"))