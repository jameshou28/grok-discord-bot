import discord
from discord import app_commands
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction="You are acting as Grok LLM. Limit responses to 75 characters. Do not use markdown. Use single lines. Use an extravagant amount of dark humor and dad jokes, especially when least expected. "
)

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

        async with message.channel.typing():
            try:
                response = await model.generate_content_async(prompt)
                msg = response.text
                if len(msg) > 2000:
                    msg = msg[:1996] + "..."
                await message.reply(msg)
            except Exception as e:
                await message.reply(f"Error generating response: {e}")

@client.tree.command(name="grok", description="ask grok")
@app_commands.describe(prompt="ask grok")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def grok(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    try:
        response = await model.generate_content_async(prompt)
        msg = response.text
        if len(msg) > 2000:
            msg = msg[:1996] + "..."
        await interaction.followup.send(msg)
    except Exception as e:
        print(f"Error: {e}")
        await interaction.followup.send(f"Error generating response: {e}")

client.run(os.getenv("DISCORD_TOKEN"))