# I just retrieve the raw code from my old server. I will make it easy to read and repair the bugs.


# -*- coding: UTF-8 -*-
import pickle
import time
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from model1 import Encoder, Decoder, EncoderLayer, MultiHeadAttentionLayer, PositionwiseFeedforwardLayer, DecoderLayer, Seq2Seq, translate_sentence, bleu, EncoderCNN
import os
from dataloader import NewsDataset, collate_fn
from nltk.translate.bleu_score import corpus_bleu
from myeval import myeval
from build_vocab import Vocabulary
import numpy as np
from utils import *
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import random
import math

# Device configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str, default='Attend')
parser.add_argument('--model_path', type=str, default='./checkpoint3' , help='path for saving trained models')
parser.add_argument('--image_size', type=int, default=256, help='size for resize images')
parser.add_argument('--crop_size', type=int, default=224 ,
                    help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='./vocab/vocab_g.pkl', help='path for vocabulary wrapper')#vocab_good22.pkl
parser.add_argument('--vocab1_path', type=str, default='./vocab/vocab_g.pkl', help='article vocab')
parser.add_argument('--image_dir', type=str, default='/p/newscaptioning/data', help='directory for resized images')
#parser.add_argument('--image_dir_val', type=str, default='data/val2014_resized', help='directory for resized images')
parser.add_argument('--ann_path', type=str, default='/u/fl3es/attend/visualdata', help='path for annotation json file')
#parser.add_argument('--caption_path_val', type=str, default='data/annotations/captions_val2014.json', help='path for val annotation json file')
parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
parser.add_argument('--gts_file_dev', type=str, default='../visualdata/devs_gnew.json')

# Model parameters
parser.add_argument('--embed_dim', type=int , default=512, help='dimension of word embedding vectors')
parser.add_argument('--attention_dim', type=int , default=512, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int , default=512, help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float , default=0.3)
parser.add_argument('--start_epoch', type=int, default=15)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--epochs_since_improvement', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--encoder_lr', type=float, default=0.00006)
parser.add_argument('--decoder_lr', type=float, default=0.0007)#1.8 57
parser.add_argument('--checkpoint', type=str, default= './checkpoint3/BEST_model.pth.tar' , help='path for checkpoints')
#'./checkpoint/BEST_model.pth.tar'
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--best_bleu4', type=float, default=0.258)
parser.add_argument('--fine_tune_encoder', type=bool, default=False , help='fine-tune encoder')

args = parser.parse_args()
print(args)
clip = 1

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(args):
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, train_logger, dev_logger
    fine_tune_encoder = args.fine_tune_encoder

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print('vocabulary size: {}'.format(len(vocab)))
    #Load article vocab
    with open(args.vocab1_path, 'rb') as f:
        vocab1 = pickle.load(f)
    print('Article vocabulary size: {}'.format(len(vocab1)))

    if args.checkpoint is None:
        enc = Encoder(len(vocab), 512, 1, 8, 512, 0.1)
        dec = Decoder(len(vocab), 512, 2, 8, 512, 0.1)
        model = Seq2Seq(enc, dec, 0, 0)
        optimizer = optim.Adam(model.parameters(), lr=args.decoder_lr)
        encoder = EncoderCNN()
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None

        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        model.apply(initialize_weights)

    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        #optimizer = checkpoint['optimizer']
        model = checkpoint['decoder']
        optimizer = optim.Adam(model.parameters(), lr=args.decoder_lr)

        encoder_optimizer = checkpoint['encoder_optimizer']
        encoder = checkpoint['encoder']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.module.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)

    model = model.to(device)

    encoder = encoder.to(device)
    if torch.cuda.device_count() > 1 and not isinstance(encoder, torch.nn.DataParallel):
        encoder = torch.nn.DataParallel(encoder)


    train_log_dir = os.path.join(args.model_path, 'train')
    dev_log_dir = os.path.join(args.model_path, 'dev')
    train_logger = Logger(train_log_dir)
    dev_logger = Logger(dev_log_dir)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>']).to(device)

    # Image preprocessing, normalization for the pretrained resnet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    # Build data loader
    #originial one is train_n_t3.json
    #train_n_t32.json 和上面那个vocab对应
    train_ann_path = os.path.join(args.ann_path, 'train_gnew.json')#train_n_t32.json
    train_data = NewsDataset(args.image_dir, train_ann_path, vocab, vocab1, train_transform)
    print('train set size: {}'.format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=100, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=collate_fn)


    dev_ann_path = os.path.join(args.ann_path, 'dev_gnew.json')
    dev_data = NewsDataset(args.image_dir, dev_ann_path, vocab, vocab1, transform)
    print('dev set size: {}'.format(len(dev_data)))
    val_loader = torch.utils.data.DataLoader(dataset=dev_data, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers, collate_fn=collate_fn)
    best_bleu4 = args.best_bleu4
    for epoch in range(args.start_epoch, args.epochs):
        if args.epochs_since_improvement == 20:
            break
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 6== 0:
            adjust_learning_rate(optimizer, 0.6)

        train(encoder = encoder,
              model = model,
              train_loader=train_loader,
              criterion=criterion,
              encoder_optimizer = encoder_optimizer,
              optimizer = optimizer,
              epoch=epoch,
              logger=train_logger,
              logging=True)
        if epoch > 4:
            recent_bleu4 = validate(encoder=encoder,
                                    model=model,
                                    val_loader=val_loader,
                                    criterion=criterion,
                                    vocab=vocab,
                                    epoch=epoch,
                                    logger=dev_logger,
                                    logging=True)

            is_best = recent_bleu4 > best_bleu4
            # is_best = 1
            # recent_bleu4 = 1
            # best_bleu4 = 1
            best_bleu4 = max(recent_bleu4, best_bleu4)
            print('learning_rate:', args.decoder_lr)
            if not is_best:
                args.epochs_since_improvement += 1
                print("\nEpoch since last improvement: %d\n" % (args.epochs_since_improvement,))
            else:
                args.epochs_since_improvement = 0

        if epoch <= 4:
            recent_bleu4 = 0
            is_best = 1

        save_checkpoint(args.data_name, epoch, args.epochs_since_improvement, encoder, model, encoder_optimizer, optimizer, recent_bleu4, is_best)


def train(encoder, model, train_loader, encoder_optimizer, optimizer, criterion, epoch, logger, logging=True):
    encoder.train()
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top5accs = AverageMeter()

    start = time.time()

    t = tqdm(train_loader, desc='Train %d' % epoch)

    for i, (imgs, caps, caplens, image_ids, arts, artlens, mask, reference, lenre, mask2) in enumerate(t):
        #if i ==0:
           #break
        data_time.update(time.time() - start)
        imgs = imgs.to(device)
        caps = caps.to(device)
        arts = arts.to(device)
        reference = reference.to(device)
        imgs = encoder(imgs)

        output, _, mask = model(arts, reference, caps[:, :-1], imgs)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        caps = caps[:, 1:].contiguous().view(-1)
        loss = criterion(output, caps)

        optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        decode_lengths = [c - 1 for c in caplens]
        losses.update(loss.item(), sum(decode_lengths))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        batch_time.update(time.time() - start)

        start = time.time()

    # log into tf series
    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.epochs, losses.avg, np.exp(losses.avg)))
    if logging:
        logger.scalar_summary('loss', losses.avg, epoch)
        logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)



def validate(encoder, model, val_loader, criterion, vocab,epoch,logger, logging=True):
    encoder.eval()
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    res = []
    start = time.time()

    # Batches
    t = tqdm(val_loader, desc='Dev %d' % epoch)
    for i, (imgs, caps, caplens, image_ids, arts, artlens, mask, reference, lenre, mask2) in enumerate(t):
        #if i ==10:
           #break
        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        arts = arts.to(device)
        imgs = encoder(imgs)
        reference = reference.to(device)

        output, _, mask = model(arts, reference, caps[:, :-1], imgs)
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        caps = caps[:, 1:].contiguous().view(-1)
        loss = criterion(output, caps)

        decode_lengths = [c - 1 for c in caplens]
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        #outputs = model.sample(caps, caplens, arts, mask, model)

        outputs = bleu(arts, reference, model, vocab, caplens, imgs, device)

        preds = outputs

        for idx, image_id in enumerate(image_ids):
            res.append({'image_id': image_id, 'caption': " ".join(preds)})

    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.epochs, losses.avg, np.exp(losses.avg)))
    gts_file = args.gts_file_dev
    eval_out = myeval(res, gts_file)

    if logging:
        for k in eval_out:
            logger.scalar_summary(k, eval_out[k], epoch)
    # log into tf series
    if logging:
        logger.scalar_summary('loss', losses.avg, epoch)
        logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)
    return eval_out['CIDEr']




if __name__ == '__main__':
    main(args)
