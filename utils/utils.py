import torch
from torch import nn
import time
import numpy as np
from torchtext.data.utils import ngrams_iterator

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    i = 0 
    for batch in iterator:
        i+=1
        trg = batch.ss
        src = batch.seq
        optimizer.zero_grad()
        
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()

        if(i%100==0):
              print(f"batch:   {i}   train_loss   {loss.item()}   perplexity   = {torch.exp(loss)}")
    
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    i = 0
    with torch.no_grad():
        for batch in iterator:
            i+=1
            trg = batch.ss
            src = batch.seq
            output = model(src, trg, 0) 
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            
            if(i%100==0):
              print(f"batch:   {i}   train_loss   {loss.item()}   perplexity   = {torch.exp(loss)}")
              
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def Predict(seq, model, vocab, ngrams, device):
    tokenizer = np.array([seq[i:i+3] for i in range(len(seq))])
    print(tokenizer)
    with torch.no_grad():
        sequence = torch.tensor([[[vocab[token]
                                for token in ngrams_iterator(tokenizer, ngrams)]]])
        sequence.to(device)
        output = model(sequence, np.array(seq.split()), torch.tensor([0]).to(device))
        return output.argmax(1).item() + 1

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs