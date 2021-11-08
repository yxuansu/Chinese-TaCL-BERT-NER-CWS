## 基于TaCL-BERT的中文命名实体识别及中文分词
基于论文 **TaCL: Improve BERT Pre-training with Token-aware Contrastive Learning**
### 环境配置
```yaml
python version == 3.8
pip install -r requirements.txt
```
### 模型结构
Chinese TaCL BERT + CRF

### Huggingface模型:

|Model Name|Model Address|
|:-------------:|:-------------:|
|Chinese (**cambridgeltl/tacl-bert-base-chinese**)|[link](https://huggingface.co/cambridgeltl/tacl-bert-base-chinese)|

### 使用范例:
```python
import torch
# initialize model
from transformers import AutoModel, AutoTokenizer
model_name = 'cambridgeltl/tacl-bert-base-chinese'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# create input ids
text = "中文TaCL-BERT模型真强大！"
text = "[CLS] " + text + " [SEP]"
tokenized_token_list = tokenizer.tokenize(text)
input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenized_token_list)).view(1, -1)
# compute hidden states
representation = model(input_ids).last_hidden_state # [1, seqlen, embed_dim]
```


## 实验
### 数据集
1. 命名实体识别: (1) MSRA (2) OntoNotes (3) Resume (4) Weibo
2. 中文分词: (1) PKU (2) CityU (3) AS

### 模型结果
|     Dataset | Precision       |Recall|F1|
| :-------------: |:-------------:|:-----:|:-----:|
|MSRA|95.41|95.47|95.44|
|OntoNotes|81.88|82.98|82.42|
|Resume|96.48|96.42|96.45|
|Weibo|68.40|70.73|69.54|
|PKU|97.04|96.46|96.75|
|CityU|98.16|98.19|98.18|
|AS|96.51|96.99|96.75|


