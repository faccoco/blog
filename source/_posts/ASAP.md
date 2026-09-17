# Paper Reading: ASAP: Attention Sink Anchored Pruning (https://arxiv.org/abs/2605.22372)

因为之前研究VGGT的时候顺带研究了很长一段时间的attention sink, 然后在看某篇神秘运控领域同名paper的时候发现了这篇,(用GPT)写了个简短的笔记记录一下

用一个“**草地上的鸟**”的例子来看：

```text
输入：CLS、草地A、草地B、鸟头、鸟身
目标：保留鸟的细节，把相似的草地合并
```

ASAP 先找一个 sink 作为参照，再看其他 token 的注意力模式与它有多相似。

下面用 NumPy 演示。**矩阵是人为构造的，代表前几层累计后的注意力关系**，不是图片像素。

```python
import numpy as np

# names：每一行、每一列对应的 token 名称。
names = np.array(["CLS", "草地A", "草地B", "鸟头", "鸟身"])

# P：累计注意力关系；一行表示一个 token 对所有 token 的关系。
P = np.array([
    # CLS   草地A  草地B  鸟头   鸟身
    [0.10, 0.60, 0.10, 0.12, 0.08],  # CLS
    [0.05, 0.70, 0.20, 0.03, 0.02],  # 草地A
    [0.05, 0.68, 0.21, 0.04, 0.02],  # 草地B
    [0.05, 0.20, 0.05, 0.50, 0.20],  # 鸟头
    [0.05, 0.25, 0.05, 0.20, 0.45],  # 鸟身
])
```

观察前两块草地：它们的整行数字几乎一样。鸟头、鸟身的模式则明显不同。

**第一步：找到 sink。** 在论文触发压缩后，它用 CLS 最关注的 patch 作为 sink。

```python
# sink：CLS 最关注的 patch 的位置。
# 跳过第 0 列 CLS，再加 1 恢复原来的位置编号。
sink = 1 + np.argmax(P[0, 1:])

print(names[sink])  # 草地A
```

**第二步：比较每个 patch 与 sink 的整行分布。** 两行越像，距离越小。

```python
# distance：四个 patch 分别与 sink 的欧氏距离。
distance = np.linalg.norm(P[1:] - P[sink], axis=1)

print(np.round(distance, 3))
# [0.000, 0.024, 0.725, 0.662]
#  草地A   草地B   鸟头   鸟身
```

草地B和 sink 很像，可以考虑合并；鸟头、鸟身与 sink 差别大，保留。

这里比较的是**整行的相似程度**，不是只比较“谁给草地A的注意力更多”。

**第三步：选出最接近 sink 的一组。** 这里把距离缩放到 0～1，再分成四段，合并第一段。

```python
# scaled：归一化距离；sink 自身距离为 0。
scaled = distance / (distance.max() + 1e-8)

# background：距离位于第一段的 patch，作为待合并组。
background = scaled < 0.25

print("合并：", names[1:][background])
print("保留：", names[1:][~background])

# 合并： ['草地A' '草地B']
# 保留： ['鸟头' '鸟身']
```

最后，**合并的是 token 的特征向量**。注意力矩阵只负责帮助我们决定合并谁。

```python
# X：示意用的 token 特征，每个 token 用两个数表示。
X = np.array([
    [0.00, 0.00],  # CLS
    [0.10, 0.20],  # 草地A
    [0.12, 0.18],  # 草地B
    [0.90, 0.80],  # 鸟头
    [0.80, 0.90],  # 鸟身
])

# weights：对待合并 token 的原始距离做 softmax，得到合并权重。
weights = np.exp(distance[background])
weights = weights / weights.sum()

# merged：两块草地的特征加权平均，生成一个代表 token。
merged = weights @ X[1:][background]

# output：保留 CLS 和鸟的 token，再加入合并后的草地 token。
output = np.vstack([X[:1], X[1:][~background], merged])
```

这段代码的实际结果是：

```text
合并后的草地特征：[0.110, 0.190]

之前：CLS、草地A、草地B、鸟头、鸟身
之后：CLS、鸟头、鸟身、草地代表
```

后续 Transformer 层就只处理这 4 个 token。实际图像中，如果有大量相似背景，减少的 token 会更多。这个例子演示的是论文的核心分组与合并步骤；“接近 sink 就适合合并”是它采用的经验判断。[ASAP 方法](https://arxiv.org/html/2605.22372v1#S3)
