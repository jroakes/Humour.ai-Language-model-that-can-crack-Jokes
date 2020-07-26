from transformers import GPT2Tokenizer


BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_LEN = 256
TRAIN_PATH = "data.csv"  #ADD PATH TO YOUR DATASET HERE
MODEL_FOLDER = "trained_models"  # ADD PATH TO WHERE YOU WANT TO SAVE YOUR MODEL
Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
