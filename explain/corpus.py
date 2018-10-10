#!/usr/bin/env python
#-*- coding: utf-8 -*-
import pickle
import torch
from torch.autograd import Variable
import math

def convert2longtensor(x):
    return torch.LongTensor(x)

def convert2variable(x, device=-1):
    if device >= 0:
        x = x.cuda(device)
    return Variable(x)


class DataGeneration(object):
    def __init__(self, vocab, data_file, batch_size=64, device=-1):
        self.batch_size = batch_size
        self.device = device
        self.data = self.load_data_file(vocab, data_file)

    def __len__(self):
        return len(self.data)

    def load_data_file(self, vocab, data_file):
        data = []
        with open(data_file) as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line) < 2:
                    continue
                src = line[0]
                trg = line[1]
#                trg = trg.split('。')[0] + ' 。'
                tmp = trg.split('。')
                for t in tmp:
                    t = t.strip()
                    if t.startswith('比喻') or t.startswith('形容'):
                        trg = t[3:] + ' 。'
                        break
                    if t.startswith('指'):
                        trg = t[2:] + '。'
                        break
                src_id = convert2longtensor(vocab.convert_to_cid(src))
                trg_id = convert2longtensor(vocab.convert_to_wid(trg.split(' ')))
                data.append((src_id, trg_id))
        return data

    def batchify(self, data):
        lens = [data[i].size(0) for i in range(len(data))]
        max_q_length = max(lens)
        text = data[0].new(len(data), max_q_length).fill_(0)

        for i in range(len(data)):
            text[i, :].narrow(0, 0, lens[i]).copy_(data[i])
        lens = convert2longtensor(lens)
        return text, lens

    def next_batch(self, shuffle=True):
        num_batch = int(math.ceil(len(self.data)/float(self.batch_size)))
        if not shuffle:
            data = self.data
            random_indexs = torch.LongTensor(range(num_batch))
        else:
            data = [self.data[i] for i in torch.randperm(len(self.data))]
            random_indexs = torch.randperm(num_batch)

        for index, i in enumerate(random_indexs):
            start, end = i * self.batch_size, (i+1) * self.batch_size
            data_tmp = data[start:end]
            src, trg = zip(*data_tmp)
            
            src_text, src_lens = self.batchify(src)
            trg_text, trg_lens = self.batchify(trg)
            src_text, src_lens, trg_text, trg_lens = [convert2variable(x, self.device) for x in [src_text, src_lens, trg_text, trg_lens]]
            yield src_text, src_lens, trg_text, trg_lens
            

if __name__ == '__main__':
    from data.dictionary import dictionary
    vocab_file = 'data/vocab.pkl'
    vocab = pickle.load(open(vocab_file, 'rb'))
    dg = DataGeneration(vocab, 'data/idiom_merge.valid')
    idx = 0
    for batch in dg.next_batch():
        idx += 1
        if idx > 1:
            break
        src_text, src_lens, trg_text, trg_lens = batch
#        print(trg_text)

