# Grok Discord Bot

A chaotic VEX robotics community bot powered by Gemini 2.5 Flash. It features real-time team scouting, semantic knowledge retrieval, and reactive meme triggers.

## Key Features
- **Team Scouting:** Non-blocking SQLite lookups for global ranks and TrueSkill scores.
- **RAG Knowledge Base:** Semantic search through `knowledge.txt` using vector embeddings.
- **Reactive Memes:** Context-aware image triggers for themes like bribery, "stupid" questions, or specific teams.
- **Smart Key Rotation:** Automatically cycles through multiple API keys to bypass quota limits.
- **Advanced Logging:** Dual-output logging to both terminal and `bot_logs.txt` with performance tracing.

## Prerequisites
- **Python 3.9+**
- A Discord Bot Token (Message Content Intent enabled).
- One or more [Google Gemini API Keys](https://aistudio.google.com/app/apikey).
- A populated `teams.db` (SQLite) and `knowledge.txt` file.

## Setup & Installation

**1. Install dependencies**:
```bash
python3 -m pip install -r requirements.txt

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
