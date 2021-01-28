
import argparse
import os
import logging
import copy
import json

import torch
from torch.utils.data import DataLoader, Dataset


from transformers import GPT2LMHeadModel

from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

from metric.ppl import Perplexity

parser = argparse.ArgumentParser(description='Evaluate KoGPT2 model')

parser.add_argument('--method',
                    type=str,
                    default='ppl',
                    help='evaluate methods')

parser.add_argument('--model_path',
                    type=str,
                    default=None,
                    help='KoGPT2 model dir')

parser.add_argument('--corpus_path',
                    type=str,
                    default=None,
                    help='path of KoGPT2 evaluation data')


parser.add_argument('--output_path',
                    type=str,
                    default=None,
                    help='path of KoGPT2 evaluation outputs')


class KoGPT2Dataset(Dataset):
    def __init__(self, datafile, max_seq_len=256):
        self.max_seq_len = max_seq_len
        # for buffer for encoder input len + 1
        self.buffed_max_seq_len = max_seq_len + 1
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.eod_token = '</s>'
        self.vocab = get_pytorch_kogpt2_model(only_vocab=True)
        tok_path = get_tokenizer()
        self.tokenizer = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)
        # logging.info(f'processing {datafile}')
        self.data = []
        try:
            logging.info(f'pre processing : {datafile}')
            with open(datafile, 'rt') as f:
                for i in f.readlines():
                    tokens = self.tokenizer(i.strip())
                    self.data.append(tokens)
        except RuntimeError as e:
            logging.warning(f'Some errors occur on {datafile}. File will be skipeed to process. {e}')
        finally:
            buffer = []
            buffed_data = []
            eod = False
            sof = True
            for sent in self.data:
                if len(sent) != 0:
                    if eod == True or sof == True:
                        eod = False
                        sof = False
                    buffer += [self.bos_token] + sent + [self.eos_token]
                #elif len(buffer) > 0:
                    #replace eos to eod
                    # buffer[-1] = self.eod_token
                if len(buffer) > self.buffed_max_seq_len:
                    buffed_data.append(buffer[:self.buffed_max_seq_len])
                    buffer = buffer[self.buffed_max_seq_len:]
            if len(buffer) > 0:
                buffed_data.append(buffer[:self.buffed_max_seq_len])
                # buffed_data[-1] += [self.eod_token]
            self.buffed_data = []
            for s in buffed_data:
                self.buffed_data.append(s)
            del buffed_data
            self.data_len = len(self.buffed_data)

    def __getitem__(self, index):
        """
        decoder in : a b c d e  f g 
        label      : b c d e f  g h 

        Args:
            index ([type]): [description]
        """
        encoder_raw = copy.deepcopy(self.buffed_data[index][:-1])
        label_raw = copy.deepcopy(self.buffed_data[index][1:])
        assert len(encoder_raw) == len(label_raw)

        decoder_in = self.vocab[encoder_raw]
        label = self.vocab[label_raw]

        att_mask = [1] * len(decoder_in)
        while len(decoder_in) < self.max_seq_len:
            decoder_in += [self.vocab['<pad>']]
            att_mask += [0]

        while len(label) < self.max_seq_len:
            label += [self.vocab['<pad>']]
        # assert len(label) == self.max_seq_len
        # assert len(decoder_in) == self.max_seq_len
        data = {'decoder_in': decoder_in[:self.max_seq_len],
                'att_mask': att_mask[:self.max_seq_len],
                'label': label[:self.max_seq_len]}

        return(data)

    def __len__(self):
        return self.data_len

    @staticmethod
    def collate_fn(batch):
        decoder_in = torch.LongTensor([item['decoder_in'] for item in batch])
        att_mask = torch.LongTensor([item['att_mask'] for item in batch])
        label = torch.LongTensor([item['label'] for item in batch])
        return {'decoder_in': decoder_in,
                'att_mask': att_mask,
                'label': label}


def corpus_level_ppl(f, model):
    dataset = KoGPT2Dataset(f, max_seq_len=1024)
    vocab = get_pytorch_kogpt2_model(only_vocab=True)
    dl = DataLoader(dataset, batch_size=7, shuffle=False, collate_fn=KoGPT2Dataset.collate_fn)
    dl = DataLoader(dataset, batch_size=7, shuffle=False, collate_fn=KoGPT2Dataset.collate_fn)
    ppls = []
    loss_avg = []
    ppl_eval = Perplexity(ignore_index=vocab['<pad>']) # pad
    total_rec = len(dl)
    reccnts = 0
    for i, batch in enumerate(dl):
        outputs = model(input_ids=batch['decoder_in'].cuda(), attention_mask=batch['att_mask'].cuda())
        ppl, loss_samp, _ = ppl_eval(outputs.logits.transpose(1, 2), batch['label'].cuda())
        ppls.append(ppl)
        loss_avg += loss_samp
        reccnts += batch['decoder_in'].shape[0]
        if i % 10 == 0:
            logging.info(f'{reccnts} of {total_rec} processed')
            ppl_avg = sum(ppls) / len(ppls)
            logging.info(f'naive ppls {ppl_avg}')
            ppl_avg_2 = Perplexity.get_perplexity(sum(loss_avg) / len(loss_avg))
            logging.info(f'ppls {ppl_avg_2}')
    return ppls, loss_avg

        

if __name__ == '__main__':
    args = parser.parse_args()
    fl = args.corpus_path
    kogpt2 = GPT2LMHeadModel.from_pretrained(args.model_path)
    kogpt2.cuda()
    kogpt2.eval()
    corpus_results = []
    logging.info('evaluating')
    logging.info(fl)
    ppl, loss_avgs = corpus_level_ppl(fl, kogpt2)
    corpus_results.append({'file': fl,
                           'naive_ppls': ppl, 
                           'losses': loss_avgs,
                           'PPL': Perplexity.get_perplexity(sum(loss_avgs) / len(loss_avgs))
                          })
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    _, tail = os.path.split(fl)
    outputfile = os.path.join(args.output_path, tail + '.json')
    json.dump(corpus_results, open(outputfile, 'wt'), indent=2)
