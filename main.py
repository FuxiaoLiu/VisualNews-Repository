
# -*- coding: UTF-8 -*-
import pickle
import time
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tqdm import tqdm
from model1261037 import EncoderCNN, AttnDecoderRNN
import os
#from data_loader import get_loader
from dataloader235 import NewsDataset, collate_fn
from nltk.translate.bleu_score import corpus_bleu
from myeval import myeval
from build_vocab import Vocabulary
import numpy as np
from utils200 import *


# Device configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='Attend')
parser.add_argument('--model_path', type=str, default='./checkpoint200' , help='path for saving trained models')
parser.add_argument('--image_size', type=int, default=256, help='size for resize images')
parser.add_argument('--crop_size', type=int, default=224 ,
                    help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='./vocab/vocab_n.pkl', help='path for vocabulary wrapper')
parser.add_argument('--vocab1_path', type=str, default='./vocab/vocab_n.pkl', help='article vocab')
parser.add_argument('--image_dir', type=str, default='/p/newscaptioning/data/WashingtonPost/images', help='directory for resized images')
#parser.add_argument('--image_dir_val', type=str, default='data/val2014_resized', help='directory for resized images')
parser.add_argument('--ann_path', type=str, default='/u/fl3es/attend', help='path for annotation json file')
#parser.add_argument('--caption_path_val', type=str, default='data/annotations/captions_val2014.json', help='path for val annotation json file')
parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
parser.add_argument('--gts_file_dev', type=str, default='./devs_n.json')

# Model parameters

parser.add_argument('--embed_dim', type=int , default=725, help='dimension of word embedding vectors')
parser.add_argument('--attention_dim', type=int , default=725, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int , default=725, help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float , default=0.3)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--epochs_since_improvement', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--encoder_lr', type=float, default=0.00004)
parser.add_argument('--decoder_lr', type=float, default=0.0007)
parser.add_argument('--checkpoint', type=str, default= None , help='path for checkpoints')
#'./checkpoint/BEST_model.pth.tar'
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--best_bleu4', type=float, default=0.)
parser.add_argument('--fine_tune_encoder', type=bool, default=False , help='fine-tune encoder')

args = parser.parse_args()
print(args)

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
    #print(vocab.word2idx['<start>'])

    if args.checkpoint is None:
        decoder = AttnDecoderRNN(attention_dim=args.attention_dim,
                                 embed_dim=args.embed_dim,
                                 decoder_dim=args.decoder_dim,
                                 vocab_size=len(vocab),
                                 vocab_size1=len(vocab1),
                                 dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),lr=args.decoder_lr)

        encoder = EncoderCNN()
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None
    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.module.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)

    decoder = decoder.to(device)
    #decoder = nn.DataParallel(decoder)

    #if torch.cuda.device_count() > 1 and not isinstance(encoder, torch.nn.DataParallel):
        #encoder = torch.nn.DataParallel(encoder)

    encoder = encoder.to(device)
    #encoder = nn.DataParallel(encoder).to(device)
    if torch.cuda.device_count() > 1 and not isinstance(encoder, torch.nn.DataParallel):
        encoder = torch.nn.DataParallel(encoder)

    train_log_dir = os.path.join(args.model_path, 'train')
    dev_log_dir = os.path.join(args.model_path, 'dev')
    train_logger = Logger(train_log_dir)
    dev_logger = Logger(dev_log_dir)

    criterion = nn.CrossEntropyLoss().to(device)

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
        #transforms.RandomCrop(args.crop_size),
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    # Build data loader
    '''
    train_loader = get_loader(args.image_dir, args.caption_path, vocab,
                              transform, args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.image_dir_val, args.caption_path_val, vocab,
                            transform, args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    '''
    train_ann_path = os.path.join(args.ann_path, 'train_n_t1.json')
    train_data = NewsDataset(args.image_dir, train_ann_path, vocab, vocab1, train_transform)
    print('train set size: {}'.format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=collate_fn)


    dev_ann_path = os.path.join(args.ann_path, 'dev_n_t1.json')
    dev_data = NewsDataset(args.image_dir, dev_ann_path, vocab, vocab1, transform)
    print('dev set size: {}'.format(len(dev_data)))
    val_loader = torch.utils.data.DataLoader(dataset=dev_data, batch_size=16, shuffle=False,
                                             num_workers=args.num_workers, collate_fn=collate_fn)
    best_bleu4 = args.best_bleu4
    for epoch in range(args.start_epoch, args.epochs):
        if epoch ==80:
            args.fine_tune_encoder = True
            encoder.module.fine_tune(args.fine_tune_encoder)
        if args.epochs_since_improvement == 20:
            break
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 6 == 0:
            adjust_learning_rate(decoder_optimizer, 0.6)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.6)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              logger=train_logger,
              logging=True)
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                vocab=vocab,
                                vocab1=vocab1,
                                epoch=epoch,
                                logger=dev_logger,
                                logging=True)

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        print('learning_rate:', args.decoder_lr)
        if not is_best:
            args.epochs_since_improvement += 1
            print("\nEpoch since last improvement: %d\n" % (args.epochs_since_improvement,))
        else:
            args.epochs_since_improvement = 0

        save_checkpoint(args.data_name, epoch, args.epochs_since_improvement, encoder, decoder,encoder_optimizer, decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,logger, logging=True):
    decoder.train()
    encoder.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top5accs = AverageMeter()

    start = time.time()

    t = tqdm(train_loader, desc='Train %d' % epoch)

    for i, (imgs, caps, caplens, image_ids, arts, artlens, mask, mask1) in enumerate(t):
        #if i ==10:
          #break
        #print('length:', caplens)
        data_time.update(time.time() - start)
        #print('caps:', caps)
        #print('length:', caplens)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        arts = arts.to(device)
        mask = mask.to(device)
        mask1 = mask1.to(device)
        imgs = encoder(imgs)

        # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        scores, caps_sorted, decode_lengths, alphas, betas = decoder(imgs, caps, caplens, arts, artlens, mask, mask1)
        #scores = scores[:, 1:]
        #alphas = alphas[:, 1:]
        #print('scores:',scores.size())
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        #print('scores:', scores)
        #print("scores:", scores)
        #print('caps_sorted:',caps_sorted.size())
        targets = caps_sorted[:, 1:]
        #print('targets:', targets.size())
        #print("target:",targets)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        #print('targets:', targets)
        #print("target1:", targets)

        loss = criterion(scores.data, targets.data)
        #loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        #loss += args.alpha_c * ((1. - betas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        #top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        #top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        #if i % args.log_step == 0:

        '''
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {:.4f}\t'
                  'Perplexity: {:5.4f}\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                          data_time=data_time, loss=losses.avg, perplexity=np.exp(losses.avg)))
        '''

    # log into tf series
    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.epochs, losses.avg, np.exp(losses.avg)))
    if logging:
        logger.scalar_summary('loss', losses.avg, epoch)
        logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)

def parse_output(outputs, vocab):

    # predicted sentences in a batch
    preds = []

    #for sentence in outputs:
    #print("+++++++")
    #print(outputs)
    '''
    for i in range(len(outputs)):
        sentence = outputs[i]
        pred = []
        print(sentence)
        print("========")
        for word in sentence:
        #for j in range(len(sentence)):
            #word = sentence[j]
            token = vocab.idx2word[word]
            if token == '<end>':
                break
            if token == '<start>':
                continue
            pred.append(token)
        preds.append(pred)
    return preds
    '''
    sentence = outputs
    for word in sentence:
        token = vocab.idx2word[word]
        if token == '<end>':
            break
        if token == '<start>':
            continue
        preds.append(token)
    return preds


def parse_output1(outputs, vocab):

    # predicted sentences in a batch
    preds = []

    for sentence in outputs:
        pred = []
        for word in sentence:
            token = vocab.idx2word[word]
            if token == '<end>':
                break
            if token == '<start>':
                continue
            pred.append(token)
        preds.append(pred)
    return preds

def validate(val_loader, encoder, decoder, criterion, vocab,vocab1,epoch,logger, logging=True):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    #top5accs = AverageMeter()
    res = []

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    t = tqdm(val_loader, desc='Dev %d' % epoch)
    for i, (imgs, caps, caplens, image_ids, arts, artlens, mask, mask1) in enumerate(t):
        #print(i)
        #if i ==40:
           #break

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        arts = arts.to(device)
        mask = mask.to(device)
        mask1 = mask1.to(device)

        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, betas = decoder(imgs, caps, caplens, arts, artlens, mask, mask1)
        #scores = scores[:, 1:]
        #alphas = alphas[:, 1:]
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets  = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores.data, targets.data)

        # Add doubly stochastic attention regularization
        #loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        #loss += args.alpha_c * ((1. - betas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        #top5 = accuracy(scores, targets, 5)
        #top5accs.update(top5, sum(decode_lengths))

        batch_time.update(time.time() - start)

        start = time.time()

        #if i % args.log_step == 0:

        '''
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss: {:.4f}\t'
                  'Perplexity: {:5.4f}\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses.avg, perplexity=np.exp(losses.avg)))
        '''
        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        #print(caps.shape)
        #print(imgs.shape)
        #print(caps.numpy())
        #print(caps)
        #print('-------')
        #print(image_ids)
        #outputs = decoder.sample1(imgs, caps, caplens, vocab)
        #outputs = decoder.sample1(imgs, caps, caplens, vocab)
        #outputs = decoder.sample1(imgs, caps, caplens, vocab, arts, artlens, vocab1)
        outputs,_ = decoder.sample1(imgs, caps, caplens, vocab, arts, artlens, vocab1, mask, mask1)
        '''
        try:
            outputs = decoder.sample1(imgs, caps, caplens, vocab,arts, artlens, vocab1)
        except:
            print("false")
            continue
        '''
        #outputs = outputs.cpu().tolist()
        #print(outputs)
        #print('gt:',caps)
        #print('output:', outputs)
        #preds = parse_output(outputs, vocab)
        outputs = outputs.cpu().tolist()
        preds = parse_output1(outputs, vocab)
        #print(preds)
        #print('preds:',preds)
        for idx, image_id in enumerate(image_ids):
            res.append({'image_id': image_id, 'caption': " ".join(preds[idx])})
    #print(res)

    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.epochs, losses.avg, np.exp(losses.avg)))
    #print('learning_rate:', )
    gts_file = args.gts_file_dev
    #print('res:',res[0])
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
