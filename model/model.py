import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import numpy as np
import random
import math
import time
from build_vocab import Vocabulary
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from build_vocab import Vocabulary
import sys
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()
        self.line = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()


    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=302):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        #self.lstm = nn.LSTM(hid_dim, hid_dim, batch_first=True)
        #self.lstm1 = nn.LSTM(hid_dim, hid_dim, batch_first=True)

    def forward(self, src, src_mask, imgs):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask, imgs)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attn_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm1 = nn.LayerNorm(hid_dim)
        self.ff_layer_norm2 = nn.LayerNorm(hid_dim)
        #self.ff_layer_norm3 = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.multimodel_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)

        #self.positionwise_feedforward1 = PositionwiseFeedforwardLayer(hid_dim,
                                                                     #pf_dim,
                                                                     #dropout)

        self.dropout = nn.Dropout(dropout)

        #self.attention1 = Attention(512, 512, 512)
        #self.attention2 = Attention(512, 512, 512)
        #self.l1 = nn.Linear(hid_dim * 2, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, src, src_mask, imgs):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        ## document inside self attention
        #print('src:', src.size())
        src0 = src
        _src, _ = self.self_attention(src, src, src, src_mask)
        #print('src:', src.size())

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        #_src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        #src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        ## pooling layer
        # mean pool or adaptive layer
        imgs1 = torch.mean(imgs, dim=1).unsqueeze(1)
        src1 = torch.mean(src, dim=1).unsqueeze(1)
        #imgs1 = self.attention1(imgs).unsqueeze(1)
        #src1 = self.attention2(src).unsqueeze(1)
        src1_ = torch.cat([imgs1, src1], dim=1)

        # multimodal attention
        src2, _ = self.multimodel_attention(src1, src1_, src1_)
        src2_ = self.tanh(self.l2(src2.expand(src2.size(0), src.size(1), src2.size(2))))
        #src2_ = self.self_attn_layer_norm1(src0 + self.dropout(src2_))

        src3 = torch.mul(src, src2_)
        _src = self.positionwise_feedforward(src)
        src4 = self.ff_layer_norm1(src + self.dropout(_src))


        #src3 = self.positionwise_feedforward1(src + src2_)
        #src4 = self.ff_layer_norm(src + self.dropout(src3))

        return src4

'''
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(encoder_out)

        att = self.full_att(self.tanh(att1)).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (att2 * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding
'''

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

        #self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.scale = np.sqrt(self.head_dim)
        self.l1 = nn.Linear(hid_dim * 2, hid_dim)
        self.l2 = nn.Linear(hid_dim * 2, hid_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.p = nn.Parameter(torch.FloatTensor(1, self.hid_dim))
        #self.init_weights()

    #def init_weights(self):
        #nn.init.normal_(self.p, 0, 1 / self.hid_dim)

    def forward(self, query, key, value, mask=None):
        #print('query:', query.size())
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

        #x = torch.matmul(self.dropout(attention), V)
        x = torch.matmul(attention, V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        x = torch.cat([x, query], dim=2)
        #print('x:', x.size())


        x1 = self.sigmoid(self.l1(x))
        x2 = self.l2(x)

        x = torch.mul(x1, x2)



        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = torch.relu(self.fc_2(x))

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
                 max_length=38):
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

        #self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.scale = np.sqrt(hid_dim)
        self.l1 = nn.Linear(hid_dim*3, 1)
        self.l2 = nn.Linear(hid_dim * 3, 1)


    def forward(self, trg, enc_src, src, trg_mask, src_mask, imgs, enc_ref, reference, ref_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        index1 = src
        index2 = reference

        #print('imgs:', imgs.size())
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        trg1 = trg
