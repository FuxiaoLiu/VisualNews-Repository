import nltk
import pickle
import argparse
from collections import Counter
import json
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(data_file, caption_type, threshold):
    #with open(data_file, 'rb') as f:
        #data = pickle.load(f)
    with open(data_file, 'r') as f:
        data = json.loads(f.read())


    max_seq_length = 0
    counter = Counter()
    print(len(data))
    #for idx, elem in enumerate(data):
    for idx in range(len(data)):
        #caption = elem[caption_type]
        caption = data[idx]['caption']
        #tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = caption.lower().split(' ')
        max_seq_length = max(len(tokens), max_seq_length)
        counter.update(tokens)
        if idx % 1000 == 0:
           print("[{}/{} captions tokenized]".format(idx, len(data)))

    print('max_seq_length: {}'.format(max_seq_length))
    #print(counter)
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = Vocabulary()
    #vocab.add_word('<unk1>')
    vocab.add_word('<pad>')  # make sure <pad> is the first token
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    for i, word in enumerate(words):
        vocab.add_word(word)
    print(words)
    return vocab

def main(args):
    vocab = build_vocab(data_file=args.caption_path,caption_type = args.caption_type, threshold=args.threshold)
    vocab_path = args.vocab_path
    print(vocab)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, default='../visualdata/train_v.json', help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./vocab/vocab_tmp.pkl',help='path for saving vocabulary wrapper')
    parser.add_argument('--caption_type', type=str, default='caption',help='caption, cleaned_caption, template_toke_coarse, template_toke_fine, \
                        compressed_caption_1, compressed_caption_2')
    parser.add_argument('--threshold', type=int, default=5000, help='minimum word count threshold')

    args = parser.parse_args()
    main(args)


