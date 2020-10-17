
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import os
import pickle
import numpy as np
import nltk
nltk.download('punkt')
from PIL import Image
import argparse
import torch
from transformers import BertTokenizer, BertModel

from build_vocab import Vocabulary

class NewsDataset(data.Dataset):

    def __init__(self, image_dir, ann_path, vocab, vocab1, transform=None, caption_type='caption'):

        self.image_dir = image_dir
        self.ann = json.load(open(ann_path, 'rb'))
        self.vocab = vocab
        self.vocab1 = vocab1
        self.transform = transform
        self.caption_type = 'caption'

    def __getitem__(self, index):

        #image part
        image_id = str(self.ann[index]['id'])
        zero = 7 - len(image_id)
        for i in range(zero):
            image_id = '0' + image_id
        file_d = image_id[:4]
        image_d = image_id[4:]
        image_path = self.image_dir + '/' + file_d + '/' + image_d + '.jpg'
        #image = Image.open(os.path.join(self.image_dir, str(image_id) + '.jpg')).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        '''
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image_id = str(self.ann[2]['id'])
            zero = 7 - len(image_id)
            for i in range(zero):
                image_id = '0' + image_id
            file_d = image_id[:4]
            image_d = image_id[4:]
            image_path = self.image_dir + '/' + file_d + '/' + image_d + '.jpg'
            image = Image.open(image_path).convert('RGB')
        '''
        if self.transform is not None:
            image = self.transform(image)
        # caption part
        # Convert caption (string) to word ids.
        # Caption
        caption = self.ann[index]['caption']
        #tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = caption.lower().split(' ')
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        #Article
        article = self.ann[index]['article']
        tokens1 = nltk.tokenize.word_tokenize(article.lower()[:500])
        article1 = []
        article1.append(self.vocab1('<start>'))
        article1.extend([self.vocab1(token1) for token1 in tokens1])
        article1.append(self.vocab1('<end>'))
        target1 = torch.Tensor(article1)
        #print(target1.size())


        '''
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_text = tokenizer.tokenize(article)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.squeeze(torch.tensor([indexed_tokens]))
        segments_tensors = torch.squeeze(torch.tensor([segments_ids]))
        '''
        #print(tokens_tensor.size())
        #print(segments_tensors.size())
        #print(target1.size())
        #print('---------')

        # Reference
        reference = self.ann[index]['article_t']
        tokens2 = nltk.tokenize.word_tokenize(reference.lower()[:300])
        reference1 = []
        reference1.append(self.vocab('<start>'))
        reference1.extend([self.vocab(token2) for token2 in tokens2])
        reference1.append(self.vocab('<end>'))
        reference1 = torch.Tensor(reference1)

        return image, target, self.ann[index]['id'], target1, reference1




    def __len__(self):
        return len(self.ann)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, articles, reference = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    #Caption
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    #print(targets.size())

    #Article
    lengths1 = [len(article) for article in articles]
    targets1 = torch.zeros(len(articles), max(lengths1)).long()
    mask1 = torch.zeros(len(articles), max(lengths1)).long()
    tmp1 = torch.ones(len(articles), max(lengths1)).long()
    for i, article in enumerate(articles):
        end = lengths1[i]
        targets1[i, :end] = article[:end]
        mask1[i, :end] = tmp1[i, :end]

    lengths2 = [len(re) for re in reference]
    targets2 = torch.zeros(len(reference), max(lengths2)).long()
    mask2 = torch.zeros(len(reference), max(lengths2)).long()
    tmp2 = torch.ones(len(reference), max(lengths2)).long()
    for i, re in enumerate(reference):
        end = lengths2[i]
        targets2[i, :end] = re[:end]
        mask2[i, :end] = tmp2[i, :end]

    return images, targets, lengths, ids, targets1, lengths1, mask1, targets2, lengths2, mask2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default='/p/newscaptioning/data/usa/images',
                        help='image directory for Reuters')
    parser.add_argument('--ann_path', type=str, default='/p/newscaptioning/data/usa/captions',
                        help='path for annotation file')

    parser.add_argument('--vocab_path', type=str, default='/p/newscaptioning/fl3es/news-captioning/src/resnet18_usa/vocab/vocab.pkl',
                        help='vocab file path')

    parser.add_argument('--caption_type', type=str, default='caption',
                        help='caption, cleaned_caption, template_toke_coarse, template_toke_fine, \
                        compressed_caption_1, compressed_caption_2')
    args = parser.parse_args()

