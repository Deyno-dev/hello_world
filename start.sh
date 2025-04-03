#!/bin/bash
# Artificial Traders v4/Multi_Ai/start.sh
python3 src/dashboard/app.py &  # Start dashboard
python3 -m src.controller      # Start controller with Telegram bot
wait