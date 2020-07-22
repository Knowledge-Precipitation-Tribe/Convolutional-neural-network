# Causal Convolution

## 因果卷积

处理序列问题（即要考虑时间问题，）就不能使用普通的CNN卷积，必须使用新的CNN模型，这个就是因果卷积的作用，看下面一个公式，对与序列问题（sequence modeling），主要抽象为，根据x1......xt和y1.....yt-1去预测yt，使得yt接近于实际值。

$$
p(\mathbf{x})=\prod_{t=1}^{T} p\left(x_{t} \mid x_{1}, \ldots, x_{t-1}\right)
$$

我们根据图片来直观理解一下因果卷积

![](../../.gitbook/assets/image%20%28122%29.png)

我们可以使用1D卷积来实现这样的因果卷积操作

```python
model.add(Conv1D(16,5, padding='causal'))
# 其中16代表多少个filter，5代表考虑多少个时间步
```

整个过程类似于下图

![](../../.gitbook/assets/image%20%28139%29.png)

黄线代表卷积核在数据的时间维度上进行移动，因为要保证时间的前后关系，不能用未来的时间数据计算当前的时间数据，所以要在数据之前添加paddind以满足时序关系。

## 膨胀因果卷积

对于因果卷积，存在的一个问题是需要很多层或者很大的filter来增加卷积的感受野。本文中，我们通过大小排列来的扩大卷积来增加感受野。扩大卷积（dilated convolution）是通过跳过部分输入来使filter可以应用于大于filter本身长度的区域。等同于通过增加零来从原始filter中生成更大的filter。

![](../../.gitbook/assets/1383870-20180730100217229-1516234421.gif)

膨胀的因果卷积我们仍然可以使用1D卷积来实现，只不过有略微差别

```python
model.add(Conv1D(16,5, padding='causal', dilation_rate=4))
```

在keras的文档中有这样一句话

> * **dilation\_rate**: an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any `dilation_rate` value != 1 is incompatible with specifying any `strides` value != 1.

文档中说dilation\_rate不等于指定卷积的步长。因为我还未看源码的具体实现，根据上面的示意图猜测是在输入数据上采用步长为2的卷积，然后在输出结果上间隔的增加padding，然后扩大一下卷积核考虑的时间片长度，得到最后的输出。

详细情况在阅读完源码后进行补充。

