# How to run and submit (simple steps)

## What is in this folder

- `bot.py` is the web server with the 5 endpoints the judge calls.
- `composer.py` writes the first message for each trigger.
- `conversation_handlers.py` handles merchant replies (auto-reply, yes, stop, off-topic).
- `llm.py` talks to Groq.
- `make_submission.py` creates `submission.jsonl` (the 30 test messages).
- `local_test.py` is a full test you can run on your laptop.
- `README.md` is the 1-page write-up to submit.

The bot works even without an API key. It then uses safe templates. With a Groq key, messages sound more natural.

## Step 1: Get a free Groq key

1. Go to console.groq.com and sign in.
2. Open "API Keys" and create a key. It starts with `gsk_`.
3. Keep it private. Never put it in GitHub.

## Step 2: Run it on your laptop

```bash
cd vera-bot
pip install -r requirements.txt
export GROQ_API_KEY=gsk_your_key      # Windows: set GROQ_API_KEY=gsk_your_key
uvicorn bot:app --port 8080
```

Open a second terminal and run:

```bash
python local_test.py
```

You will see every message the bot sends and how it replies.

## Step 3: Test with magicpin's judge

Open `judge_simulator.py` and edit the top lines:

```python
BOT_URL = "http://localhost:8080"
LLM_PROVIDER = "groq"
LLM_API_KEY = "gsk_your_key"
TEST_SCENARIO = "all"          # later try "full_evaluation"
```

Then run `python judge_simulator.py`. Restart the bot before each run so it starts fresh.

## Step 4: Make submission.jsonl again with the LLM

```bash
python make_submission.py
```

This rewrites `submission.jsonl` with LLM-polished messages. It takes about 2 minutes because it waits between calls to stay inside Groq's free limits.

## Step 5: Put it online (Render, free)

1. Push this folder to a **private** GitHub repo. Check that `.env` is not included.
2. Go to render.com and choose New, then Blueprint, and pick your repo. It reads `render.yaml`.
3. When asked, paste your `GROQ_API_KEY` and your email.
4. After deploy, open `https://<your-app>.onrender.com/v1/healthz`. You should see `"status": "ok"`.
5. Submit that base URL (without `/v1/...`) on the magicpin portal.

**Important:** Render's free plan sleeps after 15 minutes of no traffic. Waking up takes about a minute, and the bot's memory resets. Before the test window, open the healthz link to wake it. Keep it awake with a free pinger such as UptimeRobot, set to every 5 minutes. A paid plan or Railway avoids this.

A quicker option for a test: run the bot on your laptop and use `ngrok http 8080` to get a public URL. Your laptop must stay on for the whole test.

## Step 6: Before you submit

- The name (Vaibhavi Kansal) and email are already set. Change them in Render's settings only if needed.
- Read `submission.jsonl` once and fix anything that sounds odd.
- Edit `README.md` if you change the approach.
