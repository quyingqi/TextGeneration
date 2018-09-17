#!/usr/bin/env python
#-*- coding: utf-8 -*-
import utils
import torch
import pickle
from corpus import DataGeneration
from data.dictionary import dictionary

def predict_answer(model, data, vocab):
    for batch in data.next_batch(False):
        src, len_src, trg, len_trg = batch

        output = model(src, trg, len_src, teacher_forcing_ratio=0.0)

        _, output = torch.max(output, dim=2)
        output = output.data.cpu().numpy()
        src = src.data.cpu().numpy()
        for i in range(len(src)):
            a = vocab.convert_to_char(src[i])
            print(''.join(a))

            b = vocab.convert_to_word(output[i])
            print(''.join(b))
 
if __name__ == '__main__':
    args = utils.add_arguments()

    vocab = pickle.load(open(args.dict_file, 'rb'))
    valid_data = DataGeneration(vocab, args.valid_file, args.batch_size, args.device)
    model = torch.load(args.model, map_location=lambda storage, loc: storage.cuda(args.device))

    model.eval()
    predict_answer(model, valid_data, vocab)
