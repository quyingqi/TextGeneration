import os, logging, math, time, random, sys
import utils
args = utils.add_arguments()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import DataParallel
from model import Encoder, Decoder, Seq2Seq
from corpus import DataGeneration
from data.dictionary import dictionary
import pickle
import glob

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.device >= 0:
    torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)

vocab = pickle.load(open(args.dict_file, 'rb'))
args.char_size = vocab.word_size
args.word_size = vocab.char_size

print("[!] preparing dataset...")
train_data = DataGeneration(vocab, args.train_file, args.batch_size, args.device)
valid_data = DataGeneration(vocab, args.valid_file, args.batch_size, args.device)
#test_data = DataGeneration(vocab, args.test_file, args.batch_size, args.device)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, loc: storage)
    print('load model from %s' % args.resume_snapshot)
else:
    word_embedding = torch.FloatTensor(vocab.word_embedding)
    char_embedding = torch.FloatTensor(vocab.char_embedding)
    args.char_embed_size = word_embedding.size(1)
    args.word_embed_size = char_embedding.size(1)

    print("[word_vocab]:%d [char_vocab]:%d" % (args.word_size, args.char_size))

    print("[!] Instantiating models...")
    encoder = Encoder(args, word_embedding)
    decoder = Decoder(args, char_embedding)
    model = Seq2Seq(encoder, decoder)

# set model dir
model_folder, model_prefix = utils.get_folder_prefix(args, model)
log_file = model_prefix + '.log'

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

if args.multi_gpu:
    model = torch.nn.DataParallel(model)
if args.device >= 0:
    model.cuda(args.device)

params = list()
for name, param in model.named_parameters():
    log.info('%s - %s' % (name, param.size()))
    if param.requires_grad:
        params.append(param)

optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr, weight_decay=args.regular_weight)

def eval_epoch(model, data):
    model.eval()
    pad = vocab.word2id[vocab.pad]
    total_loss = 0
    for batch in data.next_batch(False):
        src, len_src, trg, len_trg = batch
        output = model(trg, src, len_trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[:, 1:, :].contiguous().view(-1, args.word_size),
                            src[:, 1:].contiguous().view(-1))
        total_loss += loss.data.item()*len(src)
    return total_loss / len(data)


def train_epoch(model, data):
    model.train()
    total_loss = 0
    pad = vocab.word2id[vocab.pad]
    batch_index = 0
    teacher_ratio = 0.5
    for batch in data.next_batch():
        batch_index += 1
#        teacher_ratio -= 0.1
        src, len_src, trg, len_trg = batch

        optimizer.zero_grad()
        output = model(trg, src, len_trg, teacher_ratio)
        loss = F.nll_loss(output[:, 1:, :].contiguous().view(-1, args.word_size),
                            src[:, 1:,].contiguous().view(-1))
        loss.backward()
        total_loss += loss.data.item()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if batch_index % 100 == 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f]" %
                  (batch_index, total_loss))
            total_loss = 0


log.info('[Training .]')
best_val_loss = None
for iter_i in range(args.epochs):
    train_epoch(model, train_data)

    val_loss = eval_epoch(model, valid_data)

    log_str = "[Epoch:%d] val_loss:%5.3f" % (iter_i, val_loss)
    log.info(log_str)

    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        print("[!] saving model...")
        single_model = model
        if isinstance(model, DataParallel):
            single_model = model.module

        for f in glob.glob(model_prefix + '*.best.loss-*'):
            os.remove(f)
        torch.save(single_model, model_prefix + '.iter-%s.best.loss-%s.model' % (iter_i, round(val_loss, 3)))
        best_val_loss = val_loss

#test_loss = eval_epoch(model, test_data)
#print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
#        main()
        predict()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
