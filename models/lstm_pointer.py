import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden, mask1 = None):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        #print('att1:', att1.size())
        #print('att2:', att2.size())
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)
        if mask1 is not None:
            att = att.masked_fill(mask1 == 0, -1e9)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class AttnDecoderRNN(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,vocab_size1, encoder_dim=512, dropout=0.3):
        super(AttnDecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.vocab_size1 = vocab_size1
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + embed_dim, decoder_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.norm1 = nn.LayerNorm(512)

        self.f1 = nn.Linear(embed_dim + encoder_dim, decoder_dim)
        self.f2 = nn.Linear(embed_dim + encoder_dim, decoder_dim)

        self.relu = nn.ReLU()

        self.embedding1 = nn.Embedding(vocab_size1, embed_dim)
        self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True, dropout=0.3)
        self.f4 = nn.Linear(embed_dim + encoder_dim, decoder_dim)
        self.f8 = nn.Linear(embed_dim + encoder_dim, decoder_dim)
        self.attention1 = Attention(encoder_dim, decoder_dim, attention_dim)
        self.attention2 = Attention(encoder_dim, decoder_dim, attention_dim)

        self.fc1 = nn.Linear(decoder_dim + decoder_dim, encoder_dim)
        self.tanh = nn.Tanh()
        self.image = nn.Linear(2048, encoder_dim)
        #self.image1 = nn.Linear(512, encoder_dim)
        self.init_weights()
        self.p = nn.Linear(decoder_dim + decoder_dim + decoder_dim, 1)


    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out, size):
        h = torch.zeros(size, 512).to(device)
        c = torch.zeros(size, 512).to(device)
        return h, c


    def init_states(self, batch_size):
        states = torch.zeros(batch_size, 512).cuda()
        return states

    def forward(self, encoder_out, encoded_captions, caption_lengths, encoded_articles, article_lengths, mask):
        """
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights
        """
        global tmp

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        embeddings1 = self.embedding1(encoded_articles)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        hiddens, _ = self.lstm(embeddings1)
        hiddens1 = self.dropout(self.relu(self.f4(hiddens)))
        #hiddens1 = hiddens + hiddens1
        #print('hiddens1:', hiddens1.size())
        hiddens2 = self.dropout(self.relu(self.f8(hiddens)))
        #hiddens = hiddens + hiddens2)


        encoder_out1 = self.dropout(self.relu(self.image(encoder_out)))
        #encoder_out2 = self.dropout(self.relu(self.image1(encoder_out)))

        hiddens = torch.cat([hiddens1, encoder_out1], dim=1)
        #hiddens2 = torch.cat([hiddens2, encoder_out2], dim=1)

        vocab_size = self.vocab_size
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out, batch_size)
        decode_lengths = [c-1 for c in caption_lengths]

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        betas = torch.zeros(batch_size, max(decode_lengths), 180).to(device)

        stack_h = []
        stack_c = []
        #h_p = h.unsqueeze(1)
        #c_p = c.unsqueeze(1)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            index = encoded_articles[:batch_size_t]

            attention_weighted_article, distribution1 = self.attention1(hiddens[:batch_size_t], h[:batch_size_t])

            h1, c1 = self.decode_step(
                torch.cat([attention_weighted_article, embeddings[:batch_size_t, t, :]], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))


            if t == 0:

                h2 = torch.cat([h1, h[:batch_size_t]], dim=1)
                h = self.dropout(self.relu(self.f1(h2)))

                c2 = torch.cat([c1, c[:batch_size_t]], dim=1)
                c = self.dropout(self.relu(self.f2(c2)))

            if t > 0:

                #c3, _ = self.attention_c(c_p[:batch_size_t], h1)
                h2 = torch.cat([h1, h3[:batch_size_t]], dim=1)
                h = self.dropout(self.relu(self.f1(h2)))


                c2 = torch.cat([c1, c3[:batch_size_t]], dim=1)
                c = self.dropout(self.relu(self.f2(c2)))


            h3 = h1
            c3 = c1

            guide_caption, distribution2 = self.attention2(hiddens2[:batch_size_t], h1, mask[:batch_size_t])
            h10 = self.tanh(self.fc1(torch.cat([guide_caption, h1], dim=1)))

            preds = self.fc(self.dropout(h10))
            preds = F.log_softmax(preds, dim=1)

            attn_value = torch.zeros([preds.size(0), preds.size(1)]).to(device)
            attn_value = attn_value.scatter_add_(1, index, distribution2)
            #attn_value = attn_value.scatter_add(1, index, distribution2)
            p = self.sigmoid(self.p(torch.cat([embeddings[:batch_size_t, t, :], guide_caption, h1[:batch_size_t]], dim=1)))
            preds = p * preds + (1-p) * attn_value

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, alphas, betas

    def sample1(self, encoder_out, caps, caplens, word_map, encoded_articles, article_lengths, voab1,mask):
        global embeddings, tmp1
        #article 部分
        batch_size = encoder_out.size(0)
        embeddings1 = self.embedding1(encoded_articles)
        encoder_out = encoder_out.reshape(encoder_out.size()[0], -1, 2048)
        hiddens, _ = self.lstm(embeddings1)

        hiddens1 = self.relu(self.f4(hiddens))
        hiddens2 = self.relu(self.f8(hiddens))

        encoder_out1 = self.relu(self.image(encoder_out))
        #encoder_out2 = self.relu(self.image1(encoder_out))

        hiddens = torch.cat([hiddens1, encoder_out1], dim=1)
        #hiddens2 = torch.cat([hiddens2, encoder_out2], dim=1)


        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * encoder_out.size()[0]).to(device)
        h, c = self.init_hidden_state(encoder_out, batch_size)
        vocab_size = self.vocab_size
        sampled_ids = []
        distribution = []
        max_seq_length = max(caplens)

        stack_h = []
        stack_c = []

        for i in range(max_seq_length):
            index = encoded_articles
            # p +=1
            if i ==0:
                embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                #h4 = h
            awa, distribution1 = self.attention1(hiddens, h)

            h1, c1 = self.decode_step(torch.cat([awa, embeddings], dim=1), (h, c))  # (s, decoder_dim)


            if i == 0:
                h2 = torch.cat([h1, h], dim=1)
                h = self.relu(self.f1(h2))

                c2 = torch.cat([c1, c], dim=1)
                c = self.relu(self.f2(c2))

            if i > 0:

                h2 = torch.cat([h1, h3], dim=1)
                h = self.relu(self.f1(h2))

                c2 = torch.cat([c1, c3], dim=1)
                c = self.relu(self.f2(c2))

            h3 = h1
            c3 = c1

            #h_p = torch.cat([h_p, h1.unsqueeze(1)], dim=1)


            guide_caption, distribution2 = self.attention2(hiddens2, h1, mask)
            h10 = self.tanh(self.fc1(torch.cat([guide_caption, h1], dim=1)))

            scores = self.fc(h10)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            attn_value = torch.zeros([scores.size(0), scores.size(1)]).to(device)
            attn_value = attn_value.scatter_add_(1, index, distribution2)
            # attn_value = attn_value.scatter_add(1, index, distribution2)
            p = self.sigmoid(self.p(torch.cat([embeddings, guide_caption, h1], dim=1)))
            scores = p * scores + (1 - p) * attn_value

            _, predicted = scores.max(1)
            sampled_ids.append(predicted)
            embeddings = self.embedding(predicted)
            embeddings = embeddings

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids, _

