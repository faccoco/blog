---
title: 从 Linear Attention 到 Test-Time Training
date: 2026-06-29 17:30:00
updated: 2026-06-29 17:30:00
categories:
  - Machine Learning
tags:
  - Attention
  - Linear Attention
  - Test-Time Training
  - Long Context
description: 从核函数线性化、递归状态和关联记忆三个视角，理解 Linear Attention 如何自然走向 Test-Time Training。
---

标准 Self-Attention 很擅长从上下文中检索信息，但它需要显式构造一个随序列长度平方增长的注意力矩阵。Linear Attention 最初试图通过核函数分解绕开这个矩阵；后来，人们发现它的递归状态还可以被理解成一个随序列更新的线性模型。沿着这个视角再走一步，就得到了 Test-Time Training（TTT）。

<!-- more -->

## 从标准注意力开始

设输入序列长度为 $n$，每个 head 的特征维度为 $d$。标准 Self-Attention 为

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{Softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V.
$$

其中 $QK^\top$ 的形状是 $n\times n$。当序列长度增加时，计算和存储这个矩阵都需要关于 $n$ 的平方复杂度。

注意力的第 $i$ 个输出也可以写成

$$
o_i=
\frac{
\sum_j
\exp(q_i^\top k_j/\sqrt d)v_j
}{
\sum_j
\exp(q_i^\top k_j/\sqrt d)
}.
$$

这里真正扮演“相似度函数”的是指数核

$$
\kappa(q,k)=\exp(q^\top k/\sqrt d).
$$

Softmax 则负责将这些相似度逐行归一化。

## 用特征映射线性化注意力

如果可以找到一个有限维特征映射 $\phi$，使

$$
\kappa(q,k)
\approx
\phi(q)^\top\phi(k),
$$

那么注意力输出就可以近似写成

$$
o_i
\approx
\frac{
\phi(q_i)^\top
\left(
\sum_j\phi(k_j)v_j^\top
\right)
}{
\phi(q_i)^\top
\left(
\sum_j\phi(k_j)
\right)
}.
$$

计算顺序由先计算全部 query-key 相似度，变成先汇总 key-value：

$$
S=\sum_j\phi(k_j)v_j^\top,
\qquad
z=\sum_j\phi(k_j).
$$

查询时只需计算

$$
o_i
\approx
\frac{\phi(q_i)^\top S}
{\phi(q_i)^\top z}.
$$

这样便不再需要显式构造 $n\times n$ 的注意力矩阵。若 $\phi$ 的维度为 $m$，value 的维度为 $d_v$，核心状态 $S$ 的大小为 $m\times d_v$，复杂度对序列长度 $n$ 是线性的。

不同工作对 $\phi$ 的处理并不相同：

- Linear Transformer 可以直接选择一个正值特征映射，例如 $\operatorname{ELU}(x)+1$，相当于使用一种可分解的相似度替代 Softmax Attention。
- Performer 使用 FAVOR+ 正交正随机特征近似 Softmax 的指数核。
- Skyformer 用 Gaussian kernel 替代 Softmax 结构，再使用 Nyström 方法近似核矩阵。

因此，“Linear Attention”不一定意味着精确复现 Softmax；更一般地，它表示注意力可以通过结合律重排，从而避免关于序列长度的平方复杂度。

## Linear Attention 的递归形式

对于因果注意力，第 $t$ 个 token 只能看到 $1,\ldots,t$。为了让后面的公式方向更直观，定义转置后的记忆矩阵

$$
W_t=
\sum_{s\leq t}
v_s\phi(k_s)^\top.
$$

它可以递归更新：

$$
W_t=W_{t-1}+
v_t\phi(k_t)^\top.
$$

同时还需要维护归一化状态

$$
z_t=z_{t-1}+
\phi(k_t).
$$

查询为

$$
o_t=
\frac{
W_t\phi(q_t)
}{
z_t^\top\phi(q_t)
}.
$$

这和前面的核注意力公式完全相同，只是从整段序列的矩阵写法变成了 RNN 式的递归写法。训练时可以使用矩阵乘法或 parallel scan 并行计算；自回归推理时则自然表现为逐 token 更新状态。

## 为什么 \(W_t\) 像一个 key-value 记忆

暂时忽略归一化项：

$$
o_t=W_t\phi(q_t).
$$

展开 $W_t$：

$$
\begin{aligned}
W_t\phi(q_t)
&=
\left(
\sum_{s\leq t}
v_s\phi(k_s)^\top
\right)\phi(q_t)\\
&=
\sum_{s\leq t}
v_s
\underbrace{
\phi(k_s)^\top\phi(q_t)
}_{\text{query 与第 }s\text{ 个 key 的相似度}}.
\end{aligned}
$$

所以 $W_t$ 存储了多组从 key 特征到 value 的关联，而 query 是读取这些关联的地址。

如果不同的 key 特征彼此正交且已经归一化，那么

$$
W_t\phi(k_i)=v_i.
$$

一般情况下 key 并不正交，因此输出是多个 value 的加权混合，而不是精确查表。这也解释了简单外积累加的一个问题：新关联只会叠加进记忆，旧关联不会被主动修改或覆盖。

## 从外积写入到 Delta Rule

无归一化 Linear Attention 的写入规则是

$$
W_t=W_{t-1}+
v_tk_t^\top,
$$

这里为了简化记号取 $\phi(x)=x$。

另一种思路是先查看当前记忆对 $k_t$ 的预测：

$$
\hat v_t=W_{t-1}k_t,
$$

然后只写入尚未预测正确的部分：

$$
W_t=W_{t-1}+
\eta_t
(v_t-W_{t-1}k_t)k_t^\top.
$$

这就是 Delta Rule。它恰好等价于对下面的平方误差执行一步梯度下降：

$$
\ell_t(W)=
\frac12
\left\|
Wk_t-v_t
\right\|^2.
$$

因为

$$
\nabla_W\ell_t(W)=
(Wk_t-v_t)k_t^\top,
$$

所以

$$
W_t=W_{t-1}-
\eta_t\nabla_W\ell_t(W_{t-1})
$$

正好得到上面的误差修正更新。

到这里，状态矩阵 $W_t$ 已经可以换一种理解：

> 它不只是一个被动累加的注意力状态，也是一个正在序列内部学习 key 到 value 映射的线性模型。

这一 fast-weight/Delta Rule 视角在 TTT 之前已经出现。TTT 的关键推进，是将它系统化为一个更一般的序列建模框架。

## Test-Time Training：让隐藏状态成为学习器

TTT 为同一个 token 构造三个由外层参数学习出的视图：

$$
k_t=\theta_Kx_t,
\qquad
v_t=\theta_Vx_t,
\qquad
q_t=\theta_Qx_t.
$$

其中：

- $k_t$ 是内部学习器的训练输入；
- $v_t$ 是内部自监督任务的学习目标；
- $q_t$ 是产生当前输出的查询输入。

TTT 将隐藏状态定义为一个模型 $f(\cdot;W_t)$ 的参数，并用当前序列提供的自监督损失更新它：

$$
\ell_t(W)=
\left\|
f(k_t;W)-v_t
\right\|^2,
$$

$$
W_t=W_{t-1}-
\eta_t
\nabla_W\ell_t(W_{t-1}).
$$

更新后使用 query view 产生输出：

$$
o_t=f(q_t;W_t).
$$

测试时，外层参数 $\theta_K,\theta_Q,\theta_V$ 通常保持不变；随输入序列临时变化的是内部状态 $W_t$。因此这里的“训练”并不是永久微调整个语言模型，而是被编入一次前向传播中的状态更新。

需要特别注意：目标

$$
Wk_t\approx v_t
$$

是 TTT 选择的内部自监督学习任务，不是 Softmax Attention 必然满足的性质。即使 $q_t=k_t$，Softmax Attention 通常仍会返回多个 value 的加权和，并不保证精确返回 $v_t$。

## Linear Attention 与 TTT 的关系

在一个特殊配置下：

- 内部模型是线性模型 $f(x;W)=Wx$；
- 初始状态 $W_0=0$；
- 对所有历史样本在同一个 $W_0$ 处计算 batch gradient；
- 平方损失的学习率取合适常数；

得到的参数为

$$
W_t=
\sum_{s\leq t}
v_sk_s^\top,
$$

输出为

$$
o_t=W_tq_t.
$$

这正是最简单的无归一化、$\phi(x)=x$ 的 Linear Attention。换句话说：

$$
\boxed{
\text{Linear Attention 是 TTT-Linear 的一个特殊情况}
}
$$

当梯度改为在当前 $W_{t-1}$ 处计算时，更新会依赖已有记忆产生的预测误差；当内部模型从线性层换成 MLP 时，就得到更具表达力的 TTT-MLP。此时模型也不再需要通过固定核函数来解释。

## 总结

从 Linear Attention 到 TTT，可以沿着下面这条路线理解：

$$
\text{Softmax 指数核}
\longrightarrow
\text{有限维特征映射}
\longrightarrow
\text{递归 KV 状态}
\longrightarrow
\text{fast-weight 线性模型}
\longrightarrow
\text{用自监督损失在线更新隐藏状态}.
$$

最初的问题是：

> 能否找到一个合适的 $\phi$，避免构造完整的注意力矩阵？

TTT 将问题进一步改写成：

> 能否把序列记忆设计成一个学习器，并让它在处理当前序列时自己学会如何压缩和读取历史？

这个视角真正扩大的是架构的搜索空间：隐藏状态不再只能是一个按固定公式更新的向量或矩阵，它还可以是一组模型参数，以及与这些参数配套的损失函数和优化器。

## 参考资料

1. Katharopoulos et al., [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html), ICML 2020.
2. Choromanski et al., [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794), ICLR 2021.
3. Schlag et al., [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html), ICML 2021.
4. Chen et al., [Skyformer: Remodel Self-Attention with Gaussian Kernel and Nyström Method](https://arxiv.org/abs/2111.00035), 2021.
5. Sun et al., [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620), 2024.
