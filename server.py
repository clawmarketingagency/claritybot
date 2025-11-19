import os
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, ApplicationBuilder

TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")   # Example: https://your-app.fly.dev/webhook

# Import your handlers
from claritybot import start, help_command, price, news, analyze, chart, signals, trend, explain, sentiment, cookie_handler

app = FastAPI()

# Build Telegram application (NO POLLING)
tg_app = ApplicationBuilder().token(TOKEN).build()

# Register handlers just like before
tg_app.add_handler(start)
tg_app.add_handler(help_command)
tg_app.add_handler(price)
tg_app.add_handler(news)
tg_app.add_handler(analyze)
tg_app.add_handler(chart)
tg_app.add_handler(signals)
tg_app.add_handler(trend)
tg_app.add_handler(explain)
tg_app.add_handler(sentiment)
tg_app.add_handler(cookie_handler)


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}


@app.on_event("startup")
async def startup_event():
    # Set Telegram webhook
    await tg_app.bot.set_webhook(url=f"{WEBHOOK_URL}")
