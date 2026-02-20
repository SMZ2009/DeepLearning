# Transformer架构
## 论文地址
https://arxiv.org/abs/1706.03762
## 概述
本文包含对于Transformer架构机制的完整描述，在本人通读论文后进行详细梳理，整理成适合初学者阅读的方式，便于Transformer架构入门。
## 前言
你需要具备的基础知识
- FNN（前馈神经网络）
- RNN（循环神经网络）
- Encoder-Decoder（编码器-解码器架构）
- Attention（注意力机制）
## 背景
### FNN
#### 解决了
- 处理固定长度输入 → 固定长度输出的问题
- 建立输入特征与输出之间的非线性映射关系