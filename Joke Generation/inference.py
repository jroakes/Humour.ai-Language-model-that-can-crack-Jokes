# Preliminaries
import os
import numpy as np 
import pandas as pd

#transformers
from transformers import GPT2LMHeadModel

# Pytorch
import torch
import torch.nn as nn

#warnings
import warnings
warnings.filterwarnings('ignore')

# My Module
import config

# HElper Function
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# Model Loading
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
special_tokens_dict = {'pad_token': '<PAD>'}
num_added_toks = config.Tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(config.Tokenizer)) 

#loading Model State
models_path = "trained_models/gpt2_medium_model{}.pt".format(int(config.EPOCHS-1)) # ADD PATH TO YOUR SAVED MODEL HERE
model.load_state_dict(torch.load(models_path))

device='cuda'
model.to(device)

def predict(length_of_joke,number_of_jokes, top_p = 0.10, top_k = 0):
    joke_num = 0
    model.eval()
    predicted_text = []

    with torch.no_grad():
        for joke_idx in range(number_of_jokes):
        
            joke_finished = False

            cur_ids = torch.tensor(config.Tokenizer.encode('SUMMARY')).unsqueeze(0).to(device)

            for i in range(length_of_joke):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                next_token_logits = logits[0, -1, :] / 0.5 # temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).item()

                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in config.Tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            
            if joke_finished:
                
                joke_num = joke_num + 1
                
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = config.Tokenizer.decode(output_list)
                output_text = output_text.replace('<|endoftext|>', '').replace('SUMMARY:', '')
                predicted_text.append(output_text)

    print(' '.join(predicted_text))

