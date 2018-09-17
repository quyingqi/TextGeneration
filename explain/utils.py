import re, os
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import torch
from torch.autograd import Variable
from argparse import ArgumentParser


def add_arguments():
    parser = ArgumentParser(description='Hyperparams')

    # Data
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-test-file', type=str, dest="test_file", default="data/idiom_merge.test")
    parser.add_argument('-train-file', type=str, dest="train_file", default="data/idiom_merge.train")
    parser.add_argument('-valid-file', type=str, dest="valid_file", default="data/idiom_merge.valid")
    parser.add_argument('-dict', type=str, dest="dict_file", default='data/vocab.pkl')
    parser.add_argument('-word_embed', type=str, default='/search/quyingqi/elmo14/tx_vocab_640k+1_vec_elmo14_ckpt_62.pkl')
    parser.add_argument('-char_embed', type=str, default='data/char_embedding.baike')

    # Train
    parser.add_argument('-epochs', type=int, default=10,
                   help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    parser.add_argument('-seed', type=int, dest="seed", default=1993)
    parser.add_argument('-exp-name', type=str, dest="exp_name", default=None, help="save model to model/$exp-name$/")
    parser.add_argument('-resume_snapshot', type=str, dest='resume_snapshot', default=None)
    parser.add_argument('-multi-gpu', action='store_true', dest='multi_gpu')
    parser.add_argument('-gpus', type=str, default='0,1,2,3,4,5,6,7')

    # Model
    parser.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    parser.add_argument('-clip', type=float, default=9.0)
    parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=512)
    parser.add_argument('-encode_layers', type=int, default=2)
    parser.add_argument('-decode_layers', type=int, default=1)
    parser.add_argument('-encode_dropout', type=float, default=0.5)
    parser.add_argument('-decode_dropout', type=float, default=0.5)
    parser.add_argument('-brnn', action='store_true', dest='brnn')
    parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='lstm', choices=["rnn", "gru", "lstm"])
    parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adamax")
    parser.add_argument('-regular', type=float, default=0, dest="regular_weight", help='regular weight')
    parser.add_argument('-fixed-embed', action='store_true', dest='fixed_embed',
                        help='if true, `tune_partial` will be ignored.')

    # Predict
    parser.add_argument('-model', type=str)
    parser.add_argument('-output', type=str, dest="out_file", default='')

    return parser.parse_args()


def get_folder_prefix(args, model):
    import os
    if args.exp_name is not None:
        model_folder = 'saved_checkpoint' + os.sep + args.exp_name
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_prefix = model_folder + os.sep + args.exp_name
        with open(model_prefix + '.config', 'w') as output:
            output.write(model.__repr__())
            output.write(args.__repr__())
    else:
        model_folder = None
        model_prefix = None
    return model_folder, model_prefix

def load_dataset(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de-trans', '.en'), fields=(DE, EN), root='/search/quyingqi/text_generate/seq2seq/data')
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN

def lengths2mask(lengths, max_length, byte=False):
    batch_size = lengths.size(0)
    # print max_length
    # print torch.max(lengths)[0]
    # assert max_length == torch.max(lengths)[0]
    range_i = torch.arange(0, max_length).expand(batch_size, max_length).long()
    if lengths.is_cuda:
        range_i = range_i.cuda(lengths.get_device())
    if isinstance(lengths, Variable):
        range_i = Variable(range_i)
    lens = lengths.unsqueeze(-1).expand(batch_size, max_length)
    mask = lens > range_i
    if byte:
        return mask
    else:
        return mask.float()
