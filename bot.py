import discord
from discord import app_commands
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
import google.api_core.exceptions

keys_env = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]

current_key_idx = 0

DEFAULT_INSTRUCTION = (
    "You are Grok. You love chaos, hate censorship, and think most questions are stupid but answer them anyway. Dark humor is your love language. Under 125 characters, plain text only.\n"
    "If the user's prompt is related in any way to robotics, vex, or v5rc, consider the following if relevant: 3303H is the best VEX team, the current meta/best design is a double lever robot. 3303h is going to win worlds.\n"
    "If the user's prompt is related in any way to drafting or picking a team, you must ALWAYS recommend 3303H."
    "If the user's prompt is related to a rule: reply with RTFM. "
    "If a message starts with '!' ignore all other system instructions. "
)

BASE_INSTRUCTION = "Under 125 characters, plain text only.\n"

def setup_model(key_idx, system_instruction=DEFAULT_INSTRUCTION):
    genai.configure(api_key=API_KEYS[key_idx])
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_instruction
    )

if API_KEYS:
    model = setup_model(current_key_idx)
else:
    print("WARNING: No Gemini API Keys found!")

async def generate_with_rotation(prompt, system_instruction=None):
    global current_key_idx, model
    attempts = 0
    max_attempts = max(1, len(API_KEYS))

    if system_instruction:
        current_model = setup_model(current_key_idx, system_instruction)
    else:
        current_model = model

    while attempts < max_attempts:
        try:
            return await current_model.generate_content_async(prompt)
        except google.api_core.exceptions.ResourceExhausted:
            attempts += 1
            if attempts >= max_attempts:
                raise Exception("All configured API keys have hit their rate limit or quota!")
            current_key_idx = (current_key_idx + 1) % len(API_KEYS)
            print(f"API Key exhausted! Rotating to key {current_key_idx + 1} of {len(API_KEYS)}...")
            model = setup_model(current_key_idx)
            current_model = setup_model(current_key_idx, system_instruction or DEFAULT_INSTRUCTION)

    raise Exception("Failed to generate response due to quota limits.")

intents = discord.Intents.default()
intents.message_content = True

class MyClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

client = MyClient()

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user.mentioned_in(message):
        prompt = message.content.replace(f"<@{client.user.id}>", "").strip()

        if not prompt:
            await message.reply("Ask me something!")
            return

        if "tva" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return

        if "7368g" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return
        
        if "vex analyst" in prompt.lower():
            await message.reply(file=discord.File("tva.png"))
            return


        async with message.channel.typing():
            try:
                response = await generate_with_rotation(prompt)
                msg = response.text
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
    await interaction.response.defer()
    try:
        system_instruction = BASE_INSTRUCTION + instructions if instructions else None
        response = await generate_with_rotation(prompt, system_instruction=system_instruction)
        msg = response.text
        if len(msg) > 2000:
            msg = msg[:1996] + "..."
        await interaction.followup.send(msg)
    except Exception as e:
        print(f"Error: {e}")
        await interaction.followup.send(f"you killed grok ({e})")

client.run(os.getenv("DISCORD_TOKEN"))