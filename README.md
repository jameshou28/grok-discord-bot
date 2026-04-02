# Grok Discord Bot

A customized Discord bot powered by Google's Gemini 2.5 Flash model (because it's free). The bot features the ability to respond to mentions and slash commands (`/grok`) with a configurable, context-aware persona. 

## Features
- **Slash Commands:** Integrated `/grok` command to quickly prompt the bot directly from Discord's command menu.
- **Mention Support:** Responds when mentioned in any channel (`@Grok <prompt>`).
- **Customizable Persona:** Easily alter the AI's internal instruction (`system_instruction`) directly in the Python code to give it a unique personality. 
- **Async Handling:** Uses optimized asynchronous Python requests to prevent the bot from crashing or freezing during heavy network usage.
- **Error Resilient:** Catches Discord's 2000-character message limit and API safety blocks, sending an alert instead of an infinite loading screen.

## Prerequisites
- **Python 3.9+** 
- A [Discord Developer](https://discord.com/developers/applications) account, a Bot Token, and an application with the `Message Content Intent` enabled.
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey).

## Setup & Installation

**1. Clone the repository** (or download the source):
   ```bash
   git clone <your-repo-url>
   cd grok
   ```

**2. Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**3. Configure Environment Variables**:
   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open your new `.env` file and insert your `DISCORD_TOKEN` and `GEMINI_API_KEY`.

**4. Adjust Bot Profile Instructions (Optional)**:
   - To alter how the AI behaves, open `bot.py` and modify the `system_instruction` text inside the `genai.GenerativeModel` initialization.
   - Note: Changing this instruction will change how the AI types responses, but it will *not* change the visual "About Me" description on Discord profiles (that must be changed in the Discord Developer Portal!).

## Running the Bot
Start the bot process using Python:
```bash
python3 bot.py
```

## Usage Menu
* `/grok <prompt>` - Ask the bot a question using Discord's built-in slash command menu.
* `@BotName <prompt>` - Standard context response via tagging the bot anywhere it can read.
