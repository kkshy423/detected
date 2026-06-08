# AHL + ADPretrain 技术节点小白解释

更新时间：2026-05-08

这份文档用尽量简单的话解释当前实验里反复出现的技术词。可以把它当作入门索引。

## 1. AHL 是什么

AHL 可以理解成“拿已经提好的图像特征来训练异常检测模型”的主流程。

它不一定直接看原图，而是更依赖 `.npy` 里的 feature。我们这次做的事情，就是把 AHL 原来吃的 DRA 特征，换成 ADPretrain 产出的特征，看效果会不会更好。

## 2. ADPretrain 是什么

ADPretrain 是另一个异常检测预训练方法。它可以从图像里提取多层视觉特征，并通过 projector 得到 residual feature。

简单说：ADPretrain 像一个新的“特征提取器”。我们想看它提出来的特征能不能替代 AHL 原来的特征。

## 3. projector 是什么

projector 是 ADPretrain 里的一个小网络模块。

它的作用是把 backbone 提取出的特征再变换一下，让正常图和异常图之间的差异更明显。

这次不是直接用原始 backbone feature，而是主要用 projector 输出后的 projected residual feature。

## 4. backbone 是什么

backbone 是视觉模型的主干网络，比如 DINOv2。

可以把它理解成“看图并提取基础视觉信息的大模型”。projector 是接在 backbone 后面的小模块。

## 5. residual feature 是什么

residual feature 可以理解成“当前图像特征”和“参考正常特征”之间的差异。

异常检测关心的是哪里不正常，所以差异特征通常比原始特征更适合表达异常。

## 6. projected residual feature 是什么

这是 residual feature 经过 projector 再处理后的结果。

当前主实验使用的就是它。原因是 ADPretrain 的设计里，projector 之后的 residual 更接近异常检测任务需要的表达。

## 7. feature 和 feature_scale 是什么

AHL 当前读取两个尺度的 `.npy`：

- `feature`：较大的空间尺度，当前是 `(512, 14, 14)`。
- `feature_scale`：较小的空间尺度，当前是 `(512, 7, 7)`。

可以理解成：一个看得更细，一个看得更粗。AHL 同时使用它们。

## 8. 512 通道是什么意思

每个空间位置不是一个数字，而是一串数字。这里这串数字长度是 512，所以叫 512 通道。

AHL 原来的头部默认适配 512 通道，因此 ADPretrain 的特征也要处理成 512 通道，才能低风险接入。

## 9. 4 层变 2 层是什么意思

ADPretrain/backbone 会输出多层特征。当前桥接规则把 4 层压成 AHL 需要的 2 个尺度：

- layer3 和 layer4 先 resize 到 `14x14`，再平均，得到 `feature`。
- layer1 和 layer2 先 resize 到 `7x7`，再平均，得到 `feature_scale`。

这样 AHL 不需要改代码，就能读新缓存。

## 10. resize 再 avg 是什么意思

不同层特征的空间大小可能不一样。不能直接相加平均。

所以先把同组特征 resize 成一样大小，再做平均。这就是“先 resize，再 avg”。

## 11. `.npy` 缓存是什么

`.npy` 是 NumPy 保存数组的文件格式。

这里每张图片会保存两个 `.npy`：

- 一个 `feature`
- 一个 `feature_scale`

AHL 训练时不重新跑大模型提特征，而是直接读取这些缓存，这样更快，也更可复现。

## 12. 为什么要训练 ADPretrain projector

官方 ADPretrain 权重不是专门为西顿数据训练的，直接用可能不适配。

所以我们用西顿 `train/good` 生成伪异常来训练 projector，让它更懂这个数据集的正常/异常差异。

## 13. 伪异常是什么

真实训练集中通常只有正常图。伪异常就是用正常图人工制造一些“看起来像异常”的变化，比如 CutPaste 风格的小块粘贴。

这样可以在不使用 test 异常的情况下训练 projector，避免数据泄漏。

## 14. 数据泄漏是什么

数据泄漏就是训练时偷偷用到了测试集信息。

如果发生数据泄漏，指标会虚高，不可信。当前 projector 训练只用 train/good，就是为了避免这个问题。

## 15. DRA baseline 是什么

baseline 是对照基线。

DRA baseline 指 AHL 原来使用 DRA 特征时的结果。它是我们要追赶或超过的目标。

当前 `models__球面斜拍` 的 DRA baseline：

- AUC-ROC：`0.9420`
- AUC-PR：`0.9621`

## 16. AUC-ROC 是什么

AUC-ROC 衡量模型把正常和异常排序分开的能力。

越接近 1 越好。0.5 接近随机猜。

在异常检测里，AUC-ROC 常用来判断整体排序能力。

## 17. AUC-PR 是什么

AUC-PR 更关注异常类的检出质量，特别适合异常样本较少的场景。

如果异常很少，AUC-PR 往往比 AUC-ROC 更敏感。

## 18. Precision、Recall、F1 是什么

- Precision：模型说是异常的样本里，有多少真的是异常。
- Recall：所有真实异常里，模型找出了多少。
- F1：Precision 和 Recall 的折中。

moment matching 后 Precision 变高、Recall 变低，说明模型更保守了：误报少了，但漏检多了一些。

## 19. moment matching 是什么

moment matching 是一个很简单的分布匹配方法。

我们发现 ADPretrain 特征的均值/方差和原 DRA 特征差很多，于是把 ADPretrain 特征整体拉到 DRA 特征的均值/方差范围。

公式是：

```text
x_out = (x - src_mean) / (src_std + eps) * ref_std + ref_mean
```

当前结果表明它有效：AUC-ROC 从 plain 的 `0.8294` 提升到 `0.8634`。

## 20. 为什么 moment matching 还没超过 DRA

因为它只修正了整体数值分布，没有改变特征本身的语义。

简单说：它能把“数值范围”调得更像 DRA，但不能保证每个通道、每个位置表达的东西和 DRA 一样。

所以下一步更合理的是 per-channel moment matching 或 residual/encoder 混合，而不是马上做更复杂的增强特征。

## 21. ASCII alias 是什么

PBS/docker 里中文路径有时会变成 `????`，导致找不到目录。

所以我们给中文类别目录建一个英文别名，比如：

- 真实类别：`models__球面斜拍`
- 实验别名：`models_qiumianxiepai`

AHL 运行时用英文别名，底层仍指向真实中文目录。

## 22. PBS 是什么

PBS 是服务器上的作业调度系统。

需要 GPU 的实验不要直接在登录节点跑，而是写 `.pbs` 文件提交，让调度系统分配 GPU 节点。

## 23. 为什么要关闭 cuDNN

AHL 的 auxiliary LSTM 在之前训练时触发了 cuDNN backward 错误。

我们没有改 AHL 源码，而是用 wrapper 在运行前设置：

```python
torch.backends.cudnn.enabled = False
```

这样绕开 cuDNN 的 LSTM 实现，训练可以稳定跑完。

## 24. 当前最重要的实验结论

链路已经跑通，但 ADPretrain 不是直接替换就能超过 DRA。

目前效果排序大致是：

```text
DRA baseline > moment matching > 西顿 projector plain > 官方 ADPretrain direct
```

这说明：

1. 西顿数据训练 projector 是必要的。
2. 特征分布匹配是有效的。
3. 还需要进一步优化特征语义或更细粒度分布匹配。

## 25. 下一步最简单的理解

不要急着做增强特征。

先把当前最有效的 moment matching 扩展到 6 个类别，看平均指标是否稳定提升。如果 6 类也有效，再做更细的 per-channel matching。