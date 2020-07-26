from transformers import GPT2Tokenizer

BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 3e-5
MAX_LEN = 256
TRAIN_PATH = "data.csv"
MODEL_FOLDER = "trained_models"
Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
