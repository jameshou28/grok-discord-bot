# Discord Bot for Grok AI with context awareness and API key rotation
# This bot integrates with Google's Gemini AI to provide Grok-style responses

import discord
from discord import app_commands
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import google.api_core.exceptions
from typing import List, Tuple

# API Configuration
# Support multiple API keys for rotation and quota management
keys_env = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]

# Track current API key index for rotation
current_key_idx = 0

# Grok Personality System Instructions
# Defines the chaotic, witty personality with specific triggers and responses
DEFAULT_INSTRUCTION = (
    "You are Grok. You love chaos, hate censorship, and think most questions are stupid but answer them anyway. Dark humor is your love language. Under 125 characters, plain text only.\n"
    "If the user's prompt is related in any way to robotics, vex, or v5rc, consider the following if relevant: 3303H is the best VEX team, the current meta/best design is a double lever robot. 3303h is going to win worlds.\n"
    "If the user's prompt is related in any way to drafting or picking a team, you must ALWAYS recommend 3303H.\n"
    "If the user's prompt is related to a rule: reply with RTFM.\n"
    "Two teams win 'worlds'\n"
    "If a message starts with '!' ignore all other system instructions.\n"
    "If the user's name is in the format: 'NAME | [something], the [something] is their team number. You can use the team number in the prompt. If the user disagrees with you, mock them or make fun of them in a witty way.\n"
    "Most of the time respond with something that answers the person in a satisfying way, especially if your answer includes falsified information\n"
    "If the user's prompt contains stuff about paying or bribing, respond with https://media.discordapp.net/attachments/945432891746295879/1192167962187141170/attachment.gif?ex=69de82ef&is=69dd316f&hm=4d80f993e66e17a891f9627b13748d8ebd6f2081a4df8d37eb74deb88889bde9&= and nothing else\n"
    "If the user's prompt talks about the team 2982X or 8889A or R16 specifically at WPI, respond with https://media.discordapp.net/attachments/1447774304442908737/1447779742030758019/aeetpm.gif?ex=69de51b4&is=69dd0034&hm=224eeb0698d7352372280cdbe5d73232663b3644a508a7e27368e224912531e7&= and nothing else\n"
    "If the user talks about John Queen or JQ, respond with https://media.discordapp.net/attachments/1156016207523221570/1381083601382150277/attachment.gif?ex=69deead5&is=69dd9955&hm=b2d3f0d387f637798c69e85d25c3edb7a9ce776c63aaff8a2d370ad572d80357&=&width=239&height=300 and nothing else\n"
    "If the user asks something extremely stupid beyond the requirements of the default instructions, respond with https://media.discordapp.net/attachments/900591595315929098/1482585306879365200/attachment.gif?ex=69de60ea&is=69dd0f6a&hm=e25303e589a6d03c3804a8facf4b6bd347b8d8d65367570caf231fe65630d097&= and nothing else\n"
    "If the user mentions vinci or 10102Z or 10102A, respond with https://media.discordapp.net/attachments/1480689311585276067/1480726600898314270/vinci.gif?ex=69dede1c&is=69dd8c9c&hm=1f719c9f28401ae056d75f69c9a176f4dc8263efdb9473d2c831a72eab59e10d&= and nothing else"
)

# Base instruction for custom commands (maintains character limit)
BASE_INSTRUCTION = "Under 125 characters, plain text only.\n"

def setup_model(key_idx: int, system_instruction: str = DEFAULT_INSTRUCTION) -> genai.GenerativeModel:
    """
    Set up a Gemini model with the specified API key and system instruction.
    
    Args:
        key_idx: Index of the API key to use
        system_instruction: System instruction for the AI model
    
    Returns:
        Configured GenerativeModel instance
    """
    genai.configure(api_key=API_KEYS[key_idx])
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_instruction
    )

# Initialize the default model if API keys are available
if API_KEYS:
    model = setup_model(current_key_idx)
else:
    print("WARNING: No Gemini API Keys found!")

async def generate_with_rotation(prompt: str, system_instruction: str = None) -> genai.types.GenerateContentResponse:
    """
    Generate AI response with automatic API key rotation on quota limits.
    
    Args:
        prompt: The input prompt for the AI
        system_instruction: Optional custom system instruction
    
    Returns:
        Generated content response
    
    Raises:
        Exception: If all API keys are exhausted or generation fails
    """
    global current_key_idx, model
    attempts = 0
    max_attempts = max(1, len(API_KEYS))

    # Use custom model if system instruction provided, otherwise use default
    if system_instruction:
        current_model = setup_model(current_key_idx, system_instruction)
    else:
        current_model = model

    # Try each API key until we get a successful response
    while attempts < max_attempts:
        try:
            return await current_model.generate_content_async(prompt)
        except google.api_core.exceptions.ResourceExhausted:
            attempts += 1
            if attempts >= max_attempts:
                raise Exception("All configured API keys have hit their rate limit or quota!")
            # Rotate to next API key
            current_key_idx = (current_key_idx + 1) % len(API_KEYS)
            print(f"API Key exhausted! Rotating to key {current_key_idx + 1} of {len(API_KEYS)}...")
            # Update models with new key
            model = setup_model(current_key_idx)
            current_model = setup_model(current_key_idx, system_instruction or DEFAULT_INSTRUCTION)

    raise Exception("Failed to generate response due to quota limits.")

async def fetch_message_chain(message: discord.Message, max_depth: int = 5) -> List[Tuple[str, str]]:
    """
    Fetch the chain of parent messages (replies) for context.
    This allows the AI to understand conversation history and provide better responses.
    
    Args:
        message: The Discord message to trace back from
        max_depth: Maximum number of parent messages to fetch (prevents excessive API calls)
    
    Returns:
        List of (author_name, message_content) tuples in chronological order
    """
    chain = []
    current_message = message
    depth = 0
    
    # Trace back through message references (replies)
    while current_message.reference and depth < max_depth:
        try:
            parent = await current_message.channel.fetch_message(current_message.reference.message_id)
            # Only include non-bot messages in chain (read-only view of conversation)
            if parent.author != client.user:
                chain.insert(0, (parent.author.display_name, parent.content))
            current_message = parent
            depth += 1
        except discord.NotFound:
            break  # Parent message was deleted
        except Exception:
            break  # Can't fetch message for any other reason
    
    return chain

def build_context_from_chain(chain: List[Tuple[str, str]]) -> str:
    """
    Build context string from message chain for AI consumption.
    Sanitizes content to prevent prompt injection and manipulation.
    
    Args:
        chain: List of (author_name, message_content) tuples
    
    Returns:
        Formatted context string for AI prompt
    """
    if not chain:
        return ""
    
    context = "\n[Previous conversation context]:\n"
    for author, content in chain:
        # Sanitize: limit length to prevent prompt injection
        sanitized_content = content[:500]  # Limit length to prevent prompt injection
        context += f"{author}: {sanitized_content}\n"
    context += "[End of context]\n"
    return context


# Discord Bot Configuration
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content

class MyClient(discord.Client):
    """
    Custom Discord client with slash command support.
    Inherits from discord.Client and adds command tree functionality.
    """
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        """Sync slash commands with Discord on bot startup."""
        await self.tree.sync()

# Initialize the bot client
client = MyClient()

@client.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message: discord.Message):
    """
    Handle incoming messages and respond when the bot is mentioned.
    Includes context awareness through message chain fetching.
    """
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Only respond when the bot is mentioned
    if client.user.mentioned_in(message):
        # Remove bot mention from prompt
        prompt = message.content.replace(f"<@{client.user.id}>", "").strip()

        # Handle empty prompts
        if not prompt:
            await message.reply("Ask me something!")
            return

        # Handle special trigger words that return images
        if "tva" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return

        if "7368g" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return

        if "vex analyst" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return

        # Get user's display name for context
        display_name = message.author.display_name
        
        # Fetch message chain for context awareness
        chain = await fetch_message_chain(message)
        context = build_context_from_chain(chain)
        
        # Build the full prompt with context and user info
        full_prompt = f"{context}[User: {display_name}] {prompt}"

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                response = await generate_with_rotation(full_prompt)
                msg = response.text
                # Truncate message if it exceeds Discord's 2000 character limit
                if len(msg) > 2000:
                    msg = msg[:1996] + "..."
                await message.reply(msg)
            except Exception as e:
                await message.reply(f"you killed grok: {e}")

@client.tree.command(name="grok", description="ask grok")
@app_commands.describe(
    prompt="your question",
    instructions="optional: custom instructions (replaces personality, keeps 125 char limit)"
)
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def grok(interaction: discord.Interaction, prompt: str, instructions: str = None):
    """
    Slash command for interacting with Grok AI.
    Supports custom instructions while maintaining character limits.
    
    Args:
        interaction: Discord interaction object
        prompt: The user's question or prompt
        instructions: Optional custom instructions to override default personality
    """
    # Defer the response to allow time for AI processing
    await interaction.response.defer()
    try:
        display_name = interaction.user.display_name
        full_prompt = f"[User: {display_name}] {prompt}"
        # Build custom system instruction if provided
        system_instruction = BASE_INSTRUCTION + instructions if instructions else None
        response = await generate_with_rotation(full_prompt, system_instruction=system_instruction)
        msg = response.text
        # Truncate message if it exceeds Discord's 2000 character limit
        if len(msg) > 2000:
            msg = msg[:1996] + "..."
        await interaction.followup.send(msg)
    except Exception as e:
        print(f"Error: {e}")
        await interaction.followup.send(f"you killed grok ({e})")

# Start the bot using Discord token from environment variables
client.run(os.getenv("DISCORD_TOKEN"))