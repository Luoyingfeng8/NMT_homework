# coding=utf-8
import torch
import random
from RNNAttention import Constant
from RNNAttention.RNNAtten import EncoderRNN, AttnDecoderRNN
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, max_len, tgt_idx2word):
    with torch.no_grad():
        input_tensor = torch.LongTensor(sentence).to(device).view(-1, 1)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[Constant.BOS]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_len, max_len)

        for di in range(max_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == Constant.EOS_WORD:
                decoded_words.append(Constant.EOS_WORD)
                break
            else:
                decoded_words.append(tgt_idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, src_id, max_len, output, tgt_idx2word):
    fout = open(output, 'w', encoding='utf-8')
    for sent in src_id:
        output_words, attentions = evaluate(encoder, decoder, sent, max_len, tgt_idx2word)
        output_sentence = ' '.join(output_words)
        fout.write('{}\n'.format(output_sentence))
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True,help='Path to model .pt file')
    parser.add_argument('-src', required=True,help='Source sequence.(one line per sequence)')
    parser.add_argument('-vocab', required=True,help='the result of preprocess.py')
    parser.add_argument('-output',required=True, help="Path to save output file")
    opt = parser.parse_args()

    with open(opt.src, 'r', encoding='utf-8') as f:
        train_src_inst = f.read().strip().split('\n')
    data = torch.load(opt.vocab)
    opt.max_len = data['setting'].max_len + 2 # eos sos
    src_word2idx = data['dict']['src']
    tgt_word2idx = data['dict']['tgt']
    tgt_idx2word = {i:word for word,i in tgt_word2idx.items()}

    src_id = [[src_word2idx.get(w, Constant.UNK) for w in s.split()] for s in train_src_inst]
    model = torch.load(opt.model)
    encoder = model['encoder']
    attn_decoder = model['decoder']
    evaluateRandomly(encoder, attn_decoder, src_id, opt.max_len, opt.output, tgt_idx2word)

if __name__ == '__main__':
    main()