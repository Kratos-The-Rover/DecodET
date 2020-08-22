import time
import random
import math

import torch
from torch import nn

from utils.utils import *
from utils.process import Iterator, Tokenizer
from utils.model import *


INPUT_DIM, OUTPUT_DIM, TRG_PAD_IDX, SEQ, SS = Tokenizer(vocab=True)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 50
CLIP = 1
BATCH_SIZE = 64


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, val_iterator, test_iterator = Iterator(device=device, batch=BATCH_SIZE)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
best_valid_loss = float('inf')


print('Starting on Device: {}\nEpochs: {}\nInput Dimension: {}\nOutput Dimension: {}\nBatch Size: {}\n\nModel:\n{}\n'.format(device, 
                                                                                                                           N_EPOCHS,
                                                                                                                           INPUT_DIM,
                                                                                                                           OUTPUT_DIM,
                                                                                                                           BATCH_SIZE,
                                                                                                                           model))

for epoch in range(N_EPOCHS):
    
    print('\nEpoch {} Started...\n'.format(epoch+1))
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print("Saving the model")
        torch.save(model.state_dict(), 'weights/'+ 'epoch_{}.pt'.format(epoch+1))
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
    
print("Saving the model after {} epochs".format(N_EPOCHS))    
torch.save(model.state_dict(), 'weights/'+ 'epoch_{}.pt'.format(N_EPOCHS))