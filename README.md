# corrector

中文错别字纠正工具。音似、形似错字（或变体字）纠正，可用于中文拼音、笔画输入法的错误纠正。python开发。


![](https://camo.githubusercontent.com/ae91a5698ad80d3fe8e0eb5a4c6ee7170e088a7d/687474703a2f2f37786b6571692e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f61692f53637265656e25323053686f74253230323031372d30342d30342532306174253230382e32302e3437253230504d2e706e67)

# Welcome

**corrector** 依据语言模型检测错别字位置，通过拼音音似特征、笔画五笔编辑距离特征及语言模型困惑度特征纠正错别字。

```
pip install -U error-correction
```

# Usage

```
from corrector import correct
line = '我们现今所使用的大部分舒学符号'
corrected_sent, correct_ranges = correct(line)
```

```
corrected_sent: 我们现今所使用的大部分数学符号
correct_ranges: [[8, 13]]
```

### 语言模型
* Kenlm（统计语言模型工具）
* RNN（TensorFlow、PaddlePaddle均有实现栈式双向LSTM的语言模型）
