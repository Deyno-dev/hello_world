# Artificial Traders v4/Multi_Ai/src/dashboard/app.py
from flask import Flask, render_template
import socketio
import logging
from pathlib import Path

app = Flask(__name__)
sio = socketio.Server()
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG_DIR / 'dashboard.log', level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@sio.event
def connect(sid, environ):
    logging.info("Client connected: %s", sid)
    sio.emit('init', {'models': ['trend', 'volatility', 'regime', 'execution', 'ensemble', 'backtrade']}, room=sid)

@sio.event
def disconnect(sid):
    logging.info("Client disconnected: %s", sid)

@sio.event
def training_update(sid, data):
    sio.emit('update', data, room=sid)

@sio.event
def training_complete(sid, data):
    sio.emit('complete', data, room=sid)

@sio.event
def error(sid, data):
    sio.emit('error', data, room=sid)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)