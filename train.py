# coding=utf-8

import torch
import time
import math
import random
import argparse
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from RNNAttention.RNNAtten import EncoderRNN, AttnDecoderRNN
from RNNAttention import Constant


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5


class MyDataset(Dataset):
    def __init__(self, src_word2idx, tgt_word2idx, src_data, tgt_data):

        assert (len(src_data) == len(src_data))

        self.src_word2idx = src_word2idx
        self.tgt_word2idx = tgt_word2idx
        self.src_idx2word = {id:word for word, id in src_word2idx.items()}
        self.tgt_idx2word = {id:word for word, id in tgt_word2idx.items()}
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab_size = len(src_word2idx)
        self.tgt_vocab_size = len(tgt_word2idx)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def build_dataloader(data):
    TrainDataset = MyDataset(
        src_word2idx=data['dict']['src'],
        tgt_word2idx=data['dict']['tgt'],
        src_data=data['train']['src'],
        tgt_data=data['train']['tgt']
    )
    ValidDataset = MyDataset(
        src_word2idx=data['dict']['src'],
        tgt_word2idx=data['dict']['tgt'],
        src_data=data['valid']['src'],
        tgt_data=data['valid']['tgt']
    )
    train_loader = DataLoader(TrainDataset, collate_fn=collate_fn_)
    valid_loader = DataLoader(ValidDataset)
    return train_loader, valid_loader

def collate_fn_(batch_data):
    x = batch_data[0][0]
    y = batch_data[0][1]
    return torch.LongTensor(x), torch.LongTensor(y)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[Constant.EOS]], device=device)


    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == Constant.EOS_WORD:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def  trainIters(encoder, decoder, training_data, epoch, save, learning_rate, max_len, print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(epoch)]

    criterion = nn.NLLLoss()

    log = open('log.out', 'w', encoding='utf-8')
    for iter in range(1, epoch + 1):
        for i ,training_pair in enumerate(training_data):
            input_tensor = training_pair[0].to(device).view(-1, 1)
            target_tensor = training_pair[1].to(device).view(-1, 1)
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len)
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('epoch:{}\ti:{}\tloss:{}'.format(epoch, i, print_loss_avg), file=log)
                log.flush()


            # if i % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0
        checkpoint = {
            'encoder': encoder,
            'decoder': decoder
        }
        model_url = '{}/checkpoint{}.pt'.format(save, iter)
        torch.save(checkpoint, model_url)
        print("Save epoch {} 's checkpoint to {}".format(iter, save))
    # showPlot(plot_losses)

def main():
    parser = argparse.ArgumentParser()
    # train parameters
    parser.add_argument('-data', required=True)
    parser.add_argument('-save', required=True, help='path to save model and log info')
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=0.01)

    # model parameters
    parser.add_argument('-hidden_size', type=int, default=512)
    opt = parser.parse_args()

    data = torch.load(opt.data)
    opt.max_len = data['setting'].max_len + 2

    training_data, validation_data = build_dataloader(data)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    encoder = EncoderRNN(opt.src_vocab_size, opt.hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(opt.hidden_size, opt.tgt_vocab_size, opt.dropout, opt.max_len).to(device)

    trainIters(encoder, attn_decoder, training_data, opt.epoch, opt.save, opt.lr, opt.max_len)

if __name__ == '__main__':
    main()






