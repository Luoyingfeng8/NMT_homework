# coding=utf-8

from RNNAttention import Constant
import argparse
import tqdm
import torch
import collections


def read_instants_pair_from_file(src_url, tgt_url, max_len, keep_case):
    src_inst = []
    tgt_inst = []
    with open(src_url, encoding='utf-8') as f1, open(tgt_url, encoding='utf-8') as f2:
        for s1, s2 in zip(f1, f2):
            s1 = s1.strip('\n')
            s2 = s2.strip('\n')
            if not keep_case:
                s1 = s1.lower()
                s2 = s2.lower()
            words1 = s1.split()
            words2 = s2.split()
            if len(words1) <= max_len and len(words2) <= max_len:
                src_inst.append([Constant.BOS_WORD] + words1 + [Constant.EOS_WORD])
                tgt_inst.append([Constant.BOS_WORD] + words2 + [Constant.EOS_WORD])
    return src_inst, tgt_inst


def build_vocab(sentence_inst, threshold):
    word2idx = {
        Constant.PAD_WORD: Constant.PAD,
        Constant.UNK_WORD: Constant.UNK,
        Constant.BOS_WORD: Constant.BOS,
        Constant.EOS_WORD: Constant.EOS
    }
    word_count = collections.defaultdict(int)
    for s in sentence_inst:
        for w in s:
            word_count[w] += 1

    ignore_count = 0
    for w , n in word_count.items():
        if w not in word2idx:
            if n > threshold:
                word2idx[w] = len(word2idx)
            else:
                ignore_count += 1
    return word2idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save', required=True)

    parser.add_argument('-max_len', type=int, default=10, help='The max length of sentence')
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-threshold', type=int, default=0, help='map words appearing less than threshold times to unk')
    opt = parser.parse_args()

    # 获取训练集和验证集句子对实例
    train_src_inst, train_tgt_inst = read_instants_pair_from_file(opt.train_src, opt.train_tgt, opt.max_len, opt.keep_case)
    valid_src_inst, valid_tgt_inst = read_instants_pair_from_file(opt.valid_src, opt.valid_tgt, opt.max_len, opt.keep_case)

    print('Get {} instances pair from trian data.'.format(len(train_src_inst)))
    print('Get {} instances pair from valid data.'.format(len(valid_src_inst)))


    # 构建单词索引表
    src_word2idx = build_vocab(train_src_inst, opt.threshold)
    tgt_word2idx = build_vocab(train_tgt_inst, opt.threshold)

    print('Source vocabulary size = {}'.format(len(src_word2idx)))
    print('Target vocabulary size = {}'.format(len(tgt_word2idx)))

    # 将训练集和验证集中的单词转化为数字
    train_src_id = [[src_word2idx.get(w, Constant.UNK) for w in s] for s in train_src_inst]
    train_tgt_id = [[tgt_word2idx.get(w, Constant.UNK) for w in s] for s in train_tgt_inst]
    valid_src_id = [[src_word2idx.get(w, Constant.UNK) for w in s] for s in valid_src_inst]
    valid_tgt_id = [[tgt_word2idx.get(w, Constant.UNK) for w in s] for s in valid_tgt_inst]

    # 保存数据
    data = {
        'setting': opt,
        'dict':{
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train':{
            'src': train_src_id,
            'tgt': train_tgt_id},
        'valid':{
            'src': valid_src_id,
            'tgt': valid_tgt_id}
    }
    print('Dumping the processed data to pickle file data.')
    torch.save(data, opt.save)
    print('Finish.')


if __name__ == '__main__':
    main()