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

training_data = {'trend': [], 'volatility': [], 'regime': [], 'execution': [], 'ensemble': []}

@app.route('/')
def index():
    return render_template('index.html')

@sio.event
def connect(sid, environ):
    logging.info("Client connected: %s", sid)
    sio.emit('init', {'models': ['trend', 'volatility', 'regime', 'execution', 'ensemble']}, room=sid)

@sio.event
def disconnect(sid):
    logging.info("Client disconnected: %s", sid)

@sio.event
def training_update(sid, data):
    model = data['model']
    training_data[model].append({'epoch': data['epoch'], 'loss': data['loss'], 'val_loss': data['val_loss']})
    # Only emit if requested
    sio.emit('update', data, room=sid)

@sio.event
def start_training(sid, data):
    action = data['action']
    if action == 'auto_train':
        logging.info("Auto training triggered from controller")
    else:
        model = action.split('_')[1]
        logging.info(f"Training {model} triggered from controller")

@sio.event
def get_training_data(sid, data):
    model = data.get('model', 'all')
    if model == 'all':
        sio.emit('training_data', training_data, room=sid)
    else:
        sio.emit('training_data', {model: training_data[model]}, room=sid)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)