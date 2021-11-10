# 基于TaCL-BERT的中文命名实体识别及中文分词
**Paper**: [TaCL: Improve BERT Pre-training with Token-aware Contrastive Learning](https://arxiv.org/abs/2111.04198)

**Authors**: Yixuan Su, Fangyu Liu, Zaiqiao Meng, Lei Shu, Ehsan Shareghi, and Nigel Collier

论文主Github repo: [https://github.com/yxuansu/TaCL](https://github.com/yxuansu/TaCL)

### 引用：
如果我们提供的资源对你有帮助，请考虑引用我们的文章。

```bibtex
@misc{su2021tacl,
      title={TaCL: Improving BERT Pre-training with Token-aware Contrastive Learning}, 
      author={Yixuan Su and Fangyu Liu and Zaiqiao Meng and Lei Shu and Ehsan Shareghi and Nigel Collier},
      year={2021},
      eprint={2111.04198},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

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
### 一、实验数据集
(1). 命名实体识别: (1) MSRA (2) OntoNotes (3) Resume (4) Weibo

(2). 中文分词: (1) PKU (2) CityU (3) AS

### 二、下载数据集
```yaml
chmod +x ./download_benchmark_data.sh
./download_benchmark_data.sh
```

### 三、下载训练好的模型
```yaml
chmod +x ./download_checkpoints.sh
./download_checkpoints.sh
```

### 四、使用训练好的模型进行inference
```yaml
cd ./sh_folder/inference/
chmod +x ./inference_{}.sh
./inference_{}.sh
```
对于不同的数据集{}的取值为['msra', 'ontonotes', 'weibo', 'resume', 'pku', 'cityu', 'as']，相关参数的含义为:

```yaml
--saved_ckpt_path: 训练好的模型位置
--train_path: 训练集数据路径
--dev_path: 验证集数据路径
--test_path: 测试集数据路径
--label_path: 数据标签路径
--batch_size: inference时的batch size
```

### 五、测试集模型结果
使用提供的模型进行inference后，可以得到如下结果。
|     Dataset | Precision       |Recall|F1|
| :-------------: |:-------------:|:-----:|:-----:|
|MSRA|95.41|95.47|95.44|
|OntoNotes|81.88|82.98|82.42|
|Resume|96.48|96.42|96.45|
|Weibo|68.40|70.73|69.54|
|PKU|97.04|96.46|96.75|
|CityU|98.16|98.19|98.18|
|AS|96.51|96.99|96.75|

### 六、从头训练一个模型
```yaml
cd ./sh_folder/train/
chmod +x ./{}.sh
./{}.sh
```
对于不同的数据集{}的取值为['msra', 'ontonotes', 'weibo', 'resume', 'pku', 'cityu', 'as']，相关参数的含义为:

```yaml
--model_name: 中文TaCL BERT的模型名称(cambridgeltl/tacl-bert-base-chinese)
--train_path: 训练集数据路径
--dev_path: 验证集数据路径
--test_path: 测试集数据路径
--label_path: 数据标签路径
--learning_rate: 学习率
--number_of_gpu: 可使用的GPU数量
--number_of_runs: 重复试验次数
--save_path_prefix: 模型存储路径
```

**[Note 1]** 我们没有对模型进行任何和学习率调参，2e-5只是默认值。通过调整学习率也许可以获得更好的结果。

**[Note 2]** 实际的batch size等于gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu。我们推荐将其设置为128。

**Inference**: 使用在./sh_folder/inference/路径中的sh进行inference。将--saved_ckpt_path设置为自己重新训练好的模型的路径。

## 交互式使用训练好的模型进行inference
以下我们使用MSRA数据集作为范例。

(**使用以下代码前，请先下载我们提供的训练好的模型以及数据集。具体的指导请见以上章节**)
```python
# 载入数据
from dataclass import Data
from transformers import AutoTokenizer
model_name = 'cambridgeltl/tacl-bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_path = r'./benchmark_data/NER/MSRANER/MSRA.test.char.txt'
label_path = r'./benchmark_data/NER/MSRANER/MSRA_NER_Label.txt'
max_len = 128
data = Data(tokenizer, data_path, data_path, data_path, label_path, max_len)

# 载入模型
import torch
from model import NERModel
model = NERModel(model_name, data.num_class)
ckpt_path = r'./pretrained_ckpt/msra/msra_ckpt'
model_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
model_parameters = model_ckpt['model']
model.load_state_dict(model_parameters)
model.eval()

# 提供输入
text = "中 共 中 央 致 中 国 致 公 党 十 一 大 的 贺 词"
text = "[CLS] " + text + " [SEP]"
tokens = tokenizer.tokenize(text)
# process token input
input_id = tokenizer.convert_tokens_to_ids(tokens)
input_id = torch.LongTensor(input_id).view(1, -1)
attn_mask = ~input_id.eq(data.pad_idx)
tgt_mask = [1.0] * len(tokens)
tgt_mask = torch.tensor(tgt_mask, dtype=torch.uint8).contiguous().view(1,-1)

# 使用模型进行解码
x = model.decode(input_id, attn_mask, tgt_mask)[0][1:-1] # remove [CLS] and [SEP] tokens.
res = ' '.join([data.id2label_dict[tag] for tag in x])
print (res)

# 模型输出结果: 
# B-NT M-NT M-NT E-NT O B-NT M-NT M-NT M-NT M-NT M-NT M-NT E-NT O O O
# 标准预测结果: 
# B-NT M-NT M-NT E-NT O B-NT M-NT M-NT M-NT M-NT M-NT M-NT E-NT O O O
```

### 联系
如果有任何的问题，以下是我的联系方式(ys484@cam.ac.uk)。
