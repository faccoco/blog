---
title: Linear attention & Test time training
date: 2026-06-29 17:30:00
updated: 2026-06-30 00:00:00
---

## Linear Attention

回忆一下self-attention 的公式

$$
Attention(X)=Softmax(\frac{QK^T}{\sqrt{d}})V
$$

$QK^T$的dim是 n x n, 这也是大部分self attention scale up不起来的原因: 随序列增长的二次方复杂度. 然而, 一个二次型是可能可以拆成两个一次运算的(核函数). 例如:

$$
(xy + c)^2
$$

可以展开为 $\phi(x)^T\phi(y)$, 其中

$$
\phi(x)=[x^2, \sqrt{2c}x, c] \\
\phi(y)=[y^2, \sqrt{2c}y, c]
$$

一个自然的想法是: self-attention中的softmax算子能不能拆分为类似的核函数? 如果能的话, 原式子能拆分成 $\frac{\phi(Q)\phi(K)^T}{\sqrt{d}}V$ → $\frac{\phi(Q)(\phi(K)^TV)}{\sqrt{d}}$, 最大的matrix shape从n x n变为了d x d (d为feature dim).

在test time training以前, 基本都是寻找各种各样的 $\phi$ 来近似softmax, 例如Efficient Attention(直接去掉softmax), Performer(线形组合特征近似), skyformer(gaussian kernel)等. 直到Test-time training的出现.

## Test-time training

在了解ttt之前, 先看看linear attention的递归形式写法

标准写法为

$$
W_t=\sum_{s\le t}v_s\phi(k_s)^\top
$$

对于一个新增的项 $t$, linear attention可以写成递归的形式.

$$
W_t=W_{t-1}+v_t\phi(k_t)^\top
$$

ttt的思路在于认为linear attention实际上是将通过attentino map进行的显式注意力图计算用一个 $d \times d$的feature 变换来估计原本需要通过 $q_tK^TV$计算出的$o_t$ . linear attention本质上是在寻找能够端到端估计$q_t$ -> $o_t$的变换$W_t$. 既然这样, 可以考虑一个在softmax注意力中合理的约束

$$
W_t k_t=v_t
$$

依据是当使用键本身进行查询($q_t=k_t$)时, $o_t = Softmax(\frac{q_tK^T}{\sqrt{d}})V$ 应当会直接返回$v_t$.  
然后使用这个约束每一个 $t$ 都对$W_t$进行推理时优化, 以此尽可能让其满足约束.
