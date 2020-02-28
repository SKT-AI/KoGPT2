# coding=utf-8
# Copyright 2020 SKT AIX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import sys

import gluonnlp as nlp
import mxnet as mx
import requests

from .model.gpt import GPT2Model as MXGPT2Model
from .utils import download as _download
from .utils import tokenizer

mxnet_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/mxnet/mxnet_kogpt2_9250bedc00.params',
    'fname': 'mxnet_kogpt2_9250bedc00.params',
    'chksum': '9250bedc00'
}


def get_mxnet_kogpt2_model(use_pooler=True,
                           use_decoder=True,
                           use_classifier=True,
                           ctx=mx.cpu(0),
                           cachedir='~/kogpt2/'):
    # download model
    model_info = mxnet_kogpt2
    model_path = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    # download vocab
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    return get_kogpt2_model(model_path, vocab_path, ctx)


def get_kogpt2_model(model_file,
                     vocab_file,
                     ctx=mx.cpu(0)):
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         mask_token=None,
                                                         sep_token=None,
                                                         cls_token=None,
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
    mxmodel = MXGPT2Model(units=768,
                          max_length=1024,
                          num_heads=12,
                          num_layers=12,
                          dropout=0.1,
                          vocab_size=len(vocab_b_obj))
    mxmodel.load_parameters(model_file, ctx=ctx)
    return (mxmodel, vocab_b_obj)
