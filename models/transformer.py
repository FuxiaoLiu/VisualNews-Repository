import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from build_vocab import Vocabulary

#import torchtext
#from torchtext.datasets import Multi30k
#from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
from build_vocab import Vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()
        self.line = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out
    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[1:]:
            for p in c.parameters():
                p.requires_grad = fine_tune




class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=33):
        super().__init__()

        #self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask,encoder_out):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask,encoder_out)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention1 = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.f = nn.Linear(512 + 512, 512)
        self.relu = nn.ReLU()

    def forward(self, trg, enc_src, trg_mask, src_mask, encoder_out):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        #_trg2, attention1 = self.encoder_attention1(trg, encoder_out, encoder_out)

        #print('_trg1:', _trg1.size())
        #print('_trg2:', _trg2.size())

        # dropout, residual connection and layer norm
        #_trg = self.relu(self.f(torch.cat([_trg1, _trg2], dim=2)))
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 vocab_size,
                 word_map):
        super().__init__()

        self.decoder = decoder
        self.src_pad_idx = word_map.word2idx['<pad>']
        self.trg_pad_idx = word_map.word2idx['<pad>']
        self.embedding1 = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True, dropout=0.3)
        self.f4 = nn.Linear(512 + 512, 512)
        self.image = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        #self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]
        #print('self.src_pad_idx:', self.src_pad_idx)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        #print('self.trg_pad_idx:', self.trg_pad_idx)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg, encoder_out):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoded_articles = src

        #print('src_mask:', src_mask)
        #print('trg_mask:', trg_mask)
        #print('+++++++++++++++++++')

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        #enc_src = self.encoder(src, src_mask)
        embeddings1 = self.embedding1(encoded_articles)
        hiddens, _ = self.lstm(embeddings1)
        enc_src = self.dropout(self.relu(self.f4(hiddens)))

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        encoder_out1 = self.dropout(self.relu(self.image(encoder_out)))


        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask, encoder_out1)
        output = F.log_softmax(output, dim=1)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention, src_mask


    def sample(self, caps, caplens, src_tensor, src_mask, encoder_out, word_map):
        #model.eval()
        # with torch.no_grad():
        # enc_src = model.encoder(src_tensor, src_mask)

        src_mask = self.make_src_mask(src_tensor)
        #print('src_tensor:', src_tensor)
        #print(src_tensor.size())

        embeddings1 = self.embedding1(src_tensor)
        hiddens, _ = self.lstm(embeddings1)
        enc_src = self.relu(self.f4(hiddens))

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        encoder_out1 = self.relu(self.image(encoder_out))
        sampled_ids = []

        # trg_indexes = [[2]] * 16
        size = src_tensor.size(0)
        trg_tensor = torch.LongTensor([[word_map.word2idx['<start>']]] * size).to(device)
        #trg_indexes = [[1]] * size
        #print('enc_src:', enc_src[:3])
        #print('trg_tensor:', trg_tensor.size())


        max_seq_length = max(caplens)
        #print('caplens:',caplens)
        for i in range(max_seq_length):
            trg_mask = self.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask, encoder_out1)
            #print('output:', output)
            #print('output:', output.size())
            pred_token = output.argmax(2)[:, -1].unsqueeze(1)
            #output1 = F.log_softmax(output, dim=1)
            _, predicted = output1.max(2)
            predicted = predicted[:, -1].unsqueeze(1)
            #print('predicted:',predicted.size())
            #print(predicted)
            #print('output1:', output1.size())
            #print(output1)
            #print('pred_token:', pred_token.size())
            #print(pred_token)
            trg_tensor = torch.cat([trg_tensor, predicted], dim=1)

            #sampled_ids.append(predicted)
            #print('trg_tensor:', trg_tensor.size())
            #print(trg_tensor)
        #sampled_ids = torch.stack(sampled_ids, 1)
        #return trg_tensor
        return trg_tensor


