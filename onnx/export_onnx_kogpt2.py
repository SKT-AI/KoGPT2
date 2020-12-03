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


import torch
import numpy as np
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
import onnxruntime


if __name__ == '__main__':
    # PATH
    MODEL_ONNX_PATH = "./onnx/pytorch_kogpt2_676e9bcfa7.onnx"
    # ONNX Settings
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
    OPSET_VERSION = 11

    tok_path = get_tokenizer()
    model, vocab = get_pytorch_kogpt2_model()
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
    sent = '2019년 한해를 보내며,'
    toked = tok(sent)

    input_ids = torch.tensor([
        vocab[vocab.bos_token],
    ] + vocab[toked]).unsqueeze(0)

    # export
    torch.onnx.export(model,
                      input_ids,
                      MODEL_ONNX_PATH,
                      verbose=True,
                      operator_export_type=OPERATOR_EXPORT_TYPE,
                      input_names=['input_ids'],
                      output_names=['output'],
                      dynamic_axes={
                          'input_ids': {
                              1: "sequence"
                          },
                          'output': {
                              1: "sequence"
                          }
                      },
                      opset_version=OPSET_VERSION)
    print("Export of kogpt2 onnx complete!")

    # inference
    sess = onnxruntime.InferenceSession(MODEL_ONNX_PATH)

    while 1:
        input_ids = torch.tensor([
            vocab[vocab.bos_token],
        ] + vocab[toked]).unsqueeze(0)
        pred = sess.run(None, {'input_ids': np.array(input_ids)})[0]
        gen = vocab.to_tokens(
            torch.argmax(torch.tensor(pred), axis=-1).squeeze().tolist())[-1]
        if gen == '</s>':
            break
        sent += gen.replace('▁', ' ')
        toked = tok(sent)
    print(sent)
