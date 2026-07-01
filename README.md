# Grok Bot

A Discord LLM assistant powered by Gemini 2.5 Flash, with semantic knowledge retrieval over a local knowledge base.

## Key Features
- **RAG Knowledge Base:** Semantic search through `knowledge.txt` using vector embeddings.
- **Smart Key Rotation:** Automatically cycles through multiple API keys to bypass quota limits.
- **Advanced Logging:** Dual-output logging to both terminal and `bot_logs.txt` with performance tracing.

## Repo layout
- `bot.py`: Discord bot entrypoint
- `knowledge.txt`: local knowledge source for RAG
- `knowledge.txt.embed_cache.pkl`: embedding cache generated on first run

## Prerequisites
- **Python 3.9+**
- A Discord Bot Token (Message Content Intent enabled).
- One or more [Google Gemini API Keys](https://aistudio.google.com/app/apikey).
- A `knowledge.txt` file.

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
