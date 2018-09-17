import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import lengths2mask


class Encoder(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    def __init__(self, args, embedding):
        super(Encoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.brnn = args.brnn
        self.rnn_type = args.rnn_type
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=args.fixed_embed)
        self.n_layers = args.encode_layers
        self.rnn = self.RNN_TYPES[args.rnn_type](args.char_embed_size, self.hidden_size, args.encode_layers,
                          dropout=args.encode_dropout, bidirectional=args.brnn, batch_first=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.rnn(embedded)
        # sum bidirectional outputs
        if self.brnn:
            outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        '''
            hidden = hidden[-2, :, :] + hidden[-1, :, :]
        else:
            hidden = hidden[-1]
        '''
        # output - (batch, len, hidden)
        # hidden - (2 * n_layers, batch, hidden)
        return outputs, hidden

class Attention_1(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs, lens):
        score = torch.bmm(encoder_outputs, hidden.unsqueeze(-1)) # [B, T, 1]
        score = score.squeeze(-1)
        if lens is not None:
            mask = lengths2mask(lens, encoder_outputs.size(1), byte=True)
            mask = ~mask
            score = score.data.masked_fill_(mask.data, float("-inf"))
        att = F.softmax(score, dim=-1)
        return att.unsqueeze(1)

class Attention_0(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs, lens):
        '''
            hidden: (batch, hidden)
            encoder_outputs: (batch, len, hidden)
        '''
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) #[B,T,H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies)

    def score(self, hidden, encoder_outputs):
        # [B,T,2H]->[B,T,H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs, lens):
        '''
            hidden: (batch, hidden)
            encoder_outputs: (batch, len, hidden)
        '''
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) #[B,T,H]
        attn_energies = self.score(h, encoder_outputs)

        if lens is not None:
            mask = lengths2mask(lens, encoder_outputs.size(1), byte=True)
            mask = ~mask
            attn_energies = attn_energies.data.masked_fill_(mask.data, float("-inf"))

        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B,T,2H]->[B,T,H]
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy.squeeze(1)


class Decoder(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    def __init__(self, args, embedding):
        super(Decoder, self).__init__()
        self.device = args.device
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=args.fixed_embed)
        self.output_size = args.word_size
        self.n_layers = args.decode_layers
        self.dropout = nn.Dropout(args.decode_dropout, inplace=True)
        self.attention = Attention(args.hidden_size)
        self.rnn = self.RNN_TYPES[args.rnn_type](args.hidden_size + args.word_embed_size, args.hidden_size,
                          args.decode_layers, dropout=args.decode_dropout, batch_first=True)
        self.out = nn.Linear(args.hidden_size * 2, args.word_size)

    def forward(self, input, last_hidden, encoder_outputs, src_len):
        '''
            input: (batch, 1)
            last_hidden: (n_layers, batch, hidden)
            encoder_outputs: (batch, len, hidden)
        '''
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(1)  # (B,1,embed)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        if len(last_hidden) == 2:
            h_n = last_hidden[0]
        else:
            h_n = last_hidden
        attn_weights = self.attention(h_n[-1, :, :], encoder_outputs, src_len) # [B,1,T]
        context = attn_weights.bmm(encoder_outputs)  # (B,1,H)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2) # (B,1,embed+H)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(1)  # (B,1,H) -> (B,H)
        context = context.squeeze(1)
        output = self.out(torch.cat([output, context], 1)) # [B,output_size]
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        batch_size, max_len = trg.size()
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(batch_size, max_len, vocab_size)).cuda(self.decoder.device)

        encoder_output, hidden = self.encoder(src)
        if self.encoder.rnn_type == 'lstm':
            hidden, c_n = hidden
            hidden = hidden[:self.decoder.n_layers]
            c_n = c_n[:self.decoder.n_layers]
            hidden = (hidden, c_n)
        else:
            hidden = hidden[:self.decoder.n_layers] # [n_layers,B,H]

        output = trg.data[:, 0] # 'sos'
        for t in range(1, max_len):
            output, hidden, attn_weights= self.decoder(
                    output, hidden, encoder_output, src_len)
            outputs[:,t,:] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = trg.data[:, t] if is_teacher else top1
        return outputs
