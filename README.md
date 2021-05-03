
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [KoGPT2 (한국어 GPT-2) Ver 2.0](#kogpt2-한국어-gpt-2-ver-20)
  - [Tokenizer](#tokenizer)
  - [Model](#model)
    - [Performances](#performances)
    - [Classification or Regression](#classification-or-regression)
  - [Data](#data)
  - [Demo](#demo)
  - [Examples](#examples)
  - [Contacts](#contacts)
  - [License](#license)

<!-- /code_chunk_output -->


## KoGPT2 (한국어 GPT-2) Ver 2.0

`GPT-2`는 주어진 텍스트의 다음 단어를 잘 예측할 수 있도록 학습된 언어모델이며 문장 생성에 최적화 되어 있습니다. `KoGPT2`는 부족한 한국어 성능을 극복하기 위해 40GB 이상의 텍스트로 학습된 한국어 디코더(`decoder`) 언어모델입니다. 

<table><tr><td>
    <center><img src="imgs/gpt2.png" width="452"/></center>
</td></tr>
</table>



### Tokenizer


[`tokenizers`](https://github.com/huggingface/tokenizers) 패키지의 `Character BPE tokenizer`로 학습되었습니다. 

사전 크기는 51,200 이며 대화에 자주 쓰이는 아래와 같은 이모티콘, 이모지 등을 추가하여 해당 토큰의 인식 능력을 올렸습니다. 
> 😀, 😁, 😆, 😅, 🤣, .. , `:-)`, `:)`, `-)`, `(-:`...

또한 `<unused0>` ~ `<unused99>`등의 미사용 토큰을 정의해 필요한 테스크에 따라 자유롭게 정의해 사용할 수 있게 했습니다.

```python
> from transformers import PreTrainedTokenizerFast
> tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
> tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o")
['▁안녕', '하', '세', '요.', '▁한국어', '▁G', 'P', 'T', '-2', '▁입', '니다.', '😤', ':)', 'l^o']
```

### Model

| Model       |  # of params |   Type   | # of layers  | # of heads | ffn_dim | hidden_dims | 
|--------------|:----:|:-------:|--------:|--------:|--------:|--------------:|
| `kogpt2-base-v2` |  125M  |  Decoder |   12     | 12      | 3072    | 768 | 


```python
> import torch
> from transformers import GPT2LMHeadModel

> model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
> text = '현대인들은 왜 항상 불안해 할까?'
> input_ids = tokenizer.encode(text)
> gen_ids = model.generate(torch.tensor([input_ids]),
                           max_length=128,
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)
> generated = tokenizer.decode(gen_ids[0,:].tolist())
> print(generated)
현대인들은 왜 항상 불안해 할까?"
"그렇다면 그건 바로 우리들의 문제입니다. 
우리가 지금 이 순간에도, 
그리고 앞으로도 계속 걱정하고 있는 것은 우리의 미래가 불투명하기 때문입니다! 
우리는 이미 오래 전부터 미래에 대한 걱정을 하고 있습니다. 그래서 ...
```

#### Performances

#### Classification or Regression

|   |  [NSMC](https://github.com/e9t/nsmc)(acc)  | [KorSTS](https://github.com/kakaobrain/KorNLUDatasets)(spearman) | PPL | 
|---|---|---|---|
| **KoGPT 2.0**  | 93.3  | 78.4  | 24.6  |
| KoGPT 1.0  | 89.9  | 80.1  | 45.4  |


### Data

한국어 위키 백과 이외, 뉴스, [모두의 말뭉치 v1.0](https://corpus.korean.go.kr/), [청와대 국민청원](https://github.com/akngs/petitions) 등의 다양한 데이터가 모델 학습에 사용되었습니다.



### Demo

[데모 링크](http://52.231.69.211:8080/)

<table><tr><td>
    <center><img src="imgs/kogpt2_2.png" width="452"/></center>
</td></tr>
</table>


### Examples

*업데이트 예정*

### Contacts

`KoGPT2` 관련 이슈는 [이곳](https://github.com/SKT-AI/KoGPT2/issues)에 올려주세요.
 

### License

`KoGPT2`는 `CC-BY-NC-SA 4.0` 라이선스 하에 공개되어 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 `LICENSE` 파일에서 확인하실 수 있습니다.
