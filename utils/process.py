from torchtext.data import Field, TabularDataset, BucketIterator
import numpy as np
import torch

def seq2ngrams(seqs, n=3):
    return np.array([seqs[i:i+n] for i in range(len(seqs))])

def Tokenizer(path='dataset/', vocab=False):
    if vocab==False:
        print("Tokenizing...")
    tokenize = lambda x: seq2ngrams(x, 3)
    tokenize_sec = lambda x: seq2ngrams(x, 1)
    seq = Field(sequential=True, use_vocab=True, tokenize=tokenize, init_token = '<sos>', eos_token = '<eos>')
    #sst3 = Field(sequential=True, use_vocab=True, tokenize=tokenize)
    sst3 = Field(sequential=True, use_vocab=True, tokenize=tokenize_sec, init_token = '<sos>', eos_token = '<eos>')
    fields = {'seq': ('seq', seq), 'sst3': ('ss', sst3)}
    
    train_data, val_data, test_data = TabularDataset.splits(
                                        path=path,
                                        train='train.csv',
                                        test='test.csv',
                                        validation='val.csv',
                                        format='csv',
                                        fields=fields)
    if vocab==False:
        print('Building Vocabulary...')
        
    seq.build_vocab(train_data,
                max_size=20000,
                min_freq=1)
    sst3.build_vocab(train_data,
                    max_size=20000,
                    min_freq=1)

    if vocab:
        return len(seq.vocab), len(sst3.vocab), sst3.vocab.stoi[sst3.pad_token], seq, sst3

    else:    
        return train_data, val_data, test_data

    
def Iterator(path='dataset/', device='cuda', batch=64):
    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
                                                    Tokenizer(path),
                                                    batch_size=batch,
                                                    device=device,
                                                    sort_key=lambda x: len(x.seq),
                                                    sort_within_batch=True)
    print('Creating batches...')
    return train_iterator, val_iterator, test_iterator