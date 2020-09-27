import torch
from torch import nn
import time
import numpy as np

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

def Predict(sequence, model, vocab, ss, ngrams, device):
    tokenizer = np.array([sequence[i:i+3] for i in range(len(sequence))])
    input_seq = []
    pred = []
    ss_pred = ''
    input_struct = []
    input_seq.append(2)
    input_struct.append(2)
    for i in range(len(tokenizer)):
        input_seq.append(vocab[tokenizer[i]])
        input_struct.append(2)
    input_seq.append(3)
    input_struct.append(2)

    input_seq_T = torch.FloatTensor(input_seq).to(torch.int64)
    input_struct_T = torch.FloatTensor(input_struct).to(torch.int64)
    input_seq_Ts = torch.unsqueeze(input_seq_T,1).to(device)
    input_strcut_Ts = torch.unsqueeze(input_struct_T,1).to(device)

    output = model(input_seq_Ts, input_strcut_Ts, 1)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)

    for j in range(1,len(output)-1):
        pred.append([k for k, v in ss.items() if v == torch.argmax(output[j]).item()])
        ss_pred+=pred[j-1][0]

    print(ss_pred) 


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs