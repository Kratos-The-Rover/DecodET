import time
import random
import math

import torch
from torch import nn

from utils.utils import Predict
from utils.process import Tokenizer
from utils.model import *


INPUT_DIM, OUTPUT_DIM, TRG_PAD_IDX, SEQ, SS = Tokenizer(vocab=True)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
CLIP = 1

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

model = Seq2Seq(enc, dec, device)
model.load_state_dict(torch.load('weights/' + 'epoch_50.pt'))
model.to(device)
sequence = input("Enter the protein sequence: ")
vocab = SEQ.vocab.stoi
ss = SS.vocab.stoi
Predict(sequence, model, vocab, ss, 1, device)