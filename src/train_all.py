# Artificial Traders v4/Multi_Ai/src/train_all.py
from .cli import main
import sys

def train_all():
    models = ['trend', 'volatility', 'regime', 'execution', 'ensemble']
    for model in models:
        print(f"Training {model}...")
        sys.argv = ['src.cli', 'train', model]
        main()

if __name__ == "__main__":
    train_all()