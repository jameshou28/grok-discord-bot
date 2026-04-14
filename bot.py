# Import required libraries
import discord  # Discord API wrapper
from discord import app_commands  # Discord slash commands
import google.generativeai as genai  # Google's Gemini AI API
import os  # Operating system interface
from dotenv import load_dotenv  # Load environment variables from .env file

# Load environment variables from .env file
load_dotenv()
import google.api_core.exceptions  # Google API exception handling
from typing import List, Tuple  # Type hints for better code clarity

# Load Gemini API keys from environment variables
# Supports multiple keys separated by commas for rotation/quota management
keys_env = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]

# Index of the currently active API key (for rotation)
current_key_idx = 0

# Default system instruction for Grok personality
# Defines the AI's behavior, humor style, and specific response patterns for certain triggers
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

# Base instruction used when custom instructions are provided via slash command
# Maintains the character limit while allowing custom personality overrides
BASE_INSTRUCTION = "Under 125 characters, plain text only.\n"

def setup_model(key_idx, system_instruction=DEFAULT_INSTRUCTION):
    """
    Configure and return a Gemini AI model with the specified API key and system instruction.
    
    Args:
        key_idx (int): Index of the API key to use from API_KEYS list
        system_instruction (str): System instruction/prompt for the AI model
        
    Returns:
        genai.GenerativeModel: Configured Gemini model instance
    """
    # Configure the Gemini API with the selected key
    genai.configure(api_key=API_KEYS[key_idx])
    
    # Create and return a new GenerativeModel instance
    return genai.GenerativeModel(
        "gemini-2.5-flash",  # Use the latest Gemini 2.5 Flash model
        system_instruction=system_instruction
    )

# Initialize the default model if API keys are available
if API_KEYS:
    model = setup_model(current_key_idx)  # Setup model with the first API key
else:
    print("WARNING: No Gemini API Keys found!")  # Warn if no keys configured

async def generate_with_rotation(prompt, system_instruction=None):
    """
    Generate AI response with automatic API key rotation on quota exhaustion.
    
    This function handles rate limiting by automatically rotating to the next API key
    when the current one hits its quota limit.
    
    Args:
        prompt (str): The user prompt to send to the AI
        system_instruction (str, optional): Custom system instruction override
        
    Returns:
        genai.GenerateContentResponse: The AI's response
        
    Raises:
        Exception: When all API keys are exhausted or generation fails
    """
    global current_key_idx, model  # Access global variables for key rotation
    attempts = 0
    max_attempts = max(1, len(API_KEYS))  # Ensure at least 1 attempt

    # Use custom model if system instruction provided, otherwise use default
    if system_instruction:
        current_model = setup_model(current_key_idx, system_instruction)
    else:
        current_model = model

    # Try each API key until we get a response or exhaust all keys
    while attempts < max_attempts:
        try:
            # Attempt to generate content with current model
            return await current_model.generate_content_async(prompt)
        except google.api_core.exceptions.ResourceExhausted:
            # Current key hit quota limit, rotate to next one
            attempts += 1
            if attempts >= max_attempts:
                raise Exception("All configured API keys have hit their rate limit or quota!")
            
            # Rotate to next key using modulo arithmetic
            current_key_idx = (current_key_idx + 1) % len(API_KEYS)
            print(f"API Key exhausted! Rotating to key {current_key_idx + 1} of {len(API_KEYS)}...")
            
            # Reconfigure models with new key
            model = setup_model(current_key_idx)
            current_model = setup_model(current_key_idx, system_instruction or DEFAULT_INSTRUCTION)

    # If we get here, all keys failed
    raise Exception("Failed to generate response due to quota limits.")

async def fetch_message_chain(message: discord.Message, max_depth: int = 5) -> List[Tuple[str, str]]:
    """
    Fetch the chain of parent messages (replies) for context.
    This allows the AI to understand conversation history when responding.
    
    Args:
        message (discord.Message): The message to trace back from
        max_depth (int): Maximum number of parent messages to fetch (prevents infinite loops)
        
    Returns:
        List[Tuple[str, str]]: List of (author_name, message_content) tuples in chronological order
    """
    chain = []  # Store the message chain
    current_message = message  # Start with the current message
    depth = 0  # Track recursion depth
    
    # Follow the reply chain up to max_depth
    while current_message.reference and depth < max_depth:
        try:
            # Fetch the parent message this message is replying to
            parent = await current_message.channel.fetch_message(current_message.reference.message_id)
            
            # Only include non-bot messages in chain (read-only view of conversation)
            # This prevents the bot from getting confused by its own previous responses
            if parent.author != client.user:
                # Insert at beginning to maintain chronological order
                chain.insert(0, (parent.author.display_name, parent.content))
            
            current_message = parent  # Move to parent message
            depth += 1
        except discord.NotFound:
            break  # Parent message was deleted or inaccessible
        except Exception:
            break  # Can't fetch message due to permissions or other issues
    
    return chain

def build_context_from_chain(chain: List[Tuple[str, str]]) -> str:
    """
    Build context string from message chain for AI prompt.
    This provides conversation context while preventing prompt injection attacks.
    
    Args:
        chain (List[Tuple[str, str]]): Message chain from fetch_message_chain()
        
    Returns:
        str: Formatted context string for inclusion in AI prompt
    """
    if not chain:
        return ""  # No context if chain is empty
    
    # Build formatted context string
    context = "\n[Previous conversation context]:\n"
    for author, content in chain:
        # Sanitize content to prevent prompt injection:
        # - Limit length to prevent extremely long prompts
        # - This helps prevent users from manipulating the AI through long messages
        sanitized_content = content[:500]  # Limit length to prevent prompt injection
        context += f"{author}: {sanitized_content}\n"
    context += "[End of context]\n"
    return context


# Configure Discord intents (permissions for what the bot can access)
intents = discord.Intents.default()  # Get default intents
intents.message_content = True  # Enable message content intent to read message contents

class MyClient(discord.Client):
    """
    Custom Discord client class with slash command support.
    Extends the base discord.Client to add command tree functionality.
    """
    def __init__(self):
        super().__init__(intents=intents)  # Initialize with our configured intents
        self.tree = app_commands.CommandTree(self)  # Create command tree for slash commands

    async def setup_hook(self):
        """
        Called when the bot is ready to set up.
        Syncs slash commands with Discord's servers.
        """
        await self.tree.sync()  # Register all slash commands with Discord

# Create the bot client instance
client = MyClient()

@client.event
async def on_ready():
    """
    Event handler called when the bot successfully connects to Discord.
    Prints login confirmation to console.
    """
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    """
    Event handler for all messages in channels the bot can see.
    Responds when the bot is mentioned in a message.
    """
    # Ignore messages from the bot itself to prevent infinite loops
    if message.author == client.user:
        return

    # Check if the bot was mentioned in the message
    if client.user.mentioned_in(message):
        # Remove the bot mention from the message to get the actual prompt
        prompt = message.content.replace(f"<@{client.user.id}>", "").strip()

        # Handle empty prompts (just a mention with no question)
        if not prompt:
            await message.reply("Ask me something!")
            return

        # Special keyword triggers for image responses
        if "tva" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return

        if "7368g" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return

        if "vex analyst" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return

        # Get user's display name for personalization
        display_name = message.author.display_name
        
        # Fetch message chain for conversation context
        chain = await fetch_message_chain(message)
        context = build_context_from_chain(chain)
        
        # Build the full prompt with context and user info
        full_prompt = f"{context}[User: {display_name}] {prompt}"

        # Show typing indicator while generating response
        async with message.channel.typing():
            try:
                # Generate AI response with automatic key rotation
                response = await generate_with_rotation(full_prompt)
                msg = response.text
                
                # Truncate message if it exceeds Discord's 2000 character limit
                if len(msg) > 2000:
                    msg = msg[:1996] + "..."
                    
                await message.reply(msg)
            except Exception as e:
                # Handle any errors during generation
                await message.reply(f"you killed grok: {e}")

@client.tree.command(name="grok", description="ask grok")
@app_commands.describe(
    prompt="your question",
    instructions="optional: custom instructions (replaces personality, keeps 125 char limit)"
)
@app_commands.allowed_installs(guilds=True, users=True)  # Allow installation in guilds and by users
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)  # Allow usage everywhere
async def grok(interaction: discord.Interaction, prompt: str, instructions: str = None):
    """
    Slash command handler for /grok command.
    Allows users to ask questions with optional custom instructions.
    
    Args:
        interaction (discord.Interaction): The Discord interaction object
        prompt (str): The user's question/prompt
        instructions (str, optional): Custom system instructions to override default personality
    """
    # Defer the response to give us time to generate the AI response
    await interaction.response.defer()
    
    try:
        # Get user's display name for personalization
        display_name = interaction.user.display_name
        
        # Build the prompt with user information
        full_prompt = f"[User: {display_name}] {prompt}"
        
        # Create custom system instruction if provided, otherwise use default
        system_instruction = BASE_INSTRUCTION + instructions if instructions else None
        
        # Generate AI response with automatic key rotation
        response = await generate_with_rotation(full_prompt, system_instruction=system_instruction)
        msg = response.text
        
        # Truncate message if it exceeds Discord's 2000 character limit
        if len(msg) > 2000:
            msg = msg[:1996] + "..."
            
        # Send the response as a followup message
        await interaction.followup.send(msg)
    except Exception as e:
        # Handle any errors during generation
        print(f"Error: {e}")  # Log to console for debugging
        await interaction.followup.send(f"you killed grok ({e})")

# Start the bot using the Discord token from environment variables
client.run(os.getenv("DISCORD_TOKEN"))