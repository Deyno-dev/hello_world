# Artificial Traders v4/Multi_Ai/src/controller.py
import subprocess
import os
import psutil
import time
from datetime import datetime
from pathlib import Path
import logging
import socketio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG_DIR / 'controller.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sio = socketio.Client()
START_TIME = time.time()
MODELS = ['trend', 'volatility', 'regime', 'execution', 'ensemble']
processes = {}

def get_system_info():
    return {
        'system_os': os.name,
        'container_os': 'ubuntu22.04',
        'cpu_usage': psutil.cpu_percent(percpu=True),
        'ram_usage': psutil.virtual_memory().percent,
        'gpu_usage': 'N/A',  # Add nvidia-smi if needed
        'cpu_temp': 'N/A'    # Requires sensors package
    }

def get_storage_info():
    models_size = {m: sum(f.stat().st_size for f in (PROJECT_ROOT / 'models' / m).glob('*')) / 1e6 for m in MODELS}
    datasets_size = sum(f.stat().st_size for f in (PROJECT_ROOT / 'data' / 'split').glob('*.csv')) / 1e6
    raw_size = sum(f.stat().st_size for f in (PROJECT_ROOT / 'data' / 'raw').glob('*.csv')) / 1e6
    return {'models': models_size, 'datasets': datasets_size, 'raw': raw_size}

def get_last_errors():
    with open(LOG_DIR / 'controller.log', 'r') as f:
        lines = [line for line in f.readlines() if 'ERROR' in line][-5:]
    return lines

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    running = {k: v.poll() is None for k, v in processes.items()}
    await update.message.reply_text(f"Status:\nTraining: {running.get('auto_train', False)}\nTrading: False (No API keys)")

async def auto_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'auto_train' not in processes or processes['auto_train'].poll() is not None:
        processes['auto_train'] = subprocess.Popen(['python', '-m', 'src.auto_train'])
        await update.message.reply_text("Started auto training")
        sio.emit('start_training', {'action': 'auto_train'})
    else:
        await update.message.reply_text("Auto training already running")

async def train_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model = context.args[0] if context.args else None
    if model not in MODELS:
        await update.message.reply_text(f"Invalid model. Use: {', '.join(MODELS)}")
        return
    if model not in processes or processes[model].poll() is not None:
        data_file = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        processes[model] = subprocess.Popen(['python', '-m', 'src.cli', 'train', model, '--data', str(data_file)])
        await update.message.reply_text(f"Started training {model}")
        sio.emit('start_training', {'action': f'train_{model}'})
    else:
        await update.message.reply_text(f"{model} training already running")

async def system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = get_system_info()
    msg = (f"System OS: {info['system_os']}\nContainer OS: {info['container_os']}\n"
           f"CPU Usage: {info['cpu_usage']}\nRAM Usage: {info['ram_usage']}%\nGPU Usage: {info['gpu_usage']}")
    await update.message.reply_text(msg)

async def uptime_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = time.time() - START_TIME
    await update.message.reply_text(f"System Uptime: {uptime / 3600:.2f} hours")

async def uptime_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("AI Uptime: Same as system (container-based)")

async def storage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = get_storage_info()
    msg = (f"Model Sizes (MB): {info['models']}\nDatasets Size (MB): {info['datasets']}\n"
           f"Raw Data Size (MB): {info['raw']}")
    await update.message.reply_text(msg)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    errors = get_last_errors()
    await update.message.reply_text("Last Errors:\n" + "\n".join(errors) if errors else "No recent errors")

def main():
    sio.connect('http://localhost:5000')
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logging.error("TELEGRAM_BOT_TOKEN not set")
        raise ValueError("Set TELEGRAM_BOT_TOKEN environment variable")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("autotrain", auto_train))
    app.add_handler(CommandHandler("train", train_model))
    app.add_handler(CommandHandler("system", system))
    app.add_handler(CommandHandler("uptimesystem", uptime_system))
    app.add_handler(CommandHandler("uptimeai", uptime_ai))
    app.add_handler(CommandHandler("storage", storage))
    app.add_handler(CommandHandler("error", error))

    logging.info("Starting Telegram bot")
    app.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Controller failed: {e}")