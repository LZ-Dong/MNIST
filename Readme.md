### 手写数字识别及数据投毒分析

#### 文件结构
- data：Mnist数据集
- classifier.py：各种分类算法函数
- train_test.py：训练和测试函数
- train_test_result.ipynb：训练和测试结果展示
- utils.py：数据集预处理函数
- poison.py：投毒函数
- poison.ipynb：投毒结果展示

#### utils.py
使用示例
```
from utils import *
image = decode_idx3_ubyte('./data/train-images.idx3-ubyte')
label = decode_idx1_ubyte('./data/train-labels.idx1-ubyte')
```
我使用的ide是vscode，文件路径请以自己使用的ide为准。
image：数据集数量\*28像素\*28像素的array。
label：数据集数量的array。