# Causal Convolution

## 因果卷积

处理序列问题（即要考虑时间问题，）就不能使用普通的CNN卷积，必须使用新的CNN模型，这个就是因果卷积的作用，看下面一个公式，对与序列问题（sequence modeling），主要抽象为，根据x1......xt和y1.....yt-1去预测yt，使得yt接近于实际值。

$$
p(\mathbf{x})=\prod_{t=1}^{T} p\left(x_{t} \mid x_{1}, \ldots, x_{t-1}\right)
$$

我们根据图片来直观理解一下因果卷积

![](../../.gitbook/assets/image%20%28122%29.png)

## 膨胀因果卷积

对于因果卷积，存在的一个问题是需要很多层或者很大的filter来增加卷积的感受野。本文中，我们通过大小排列来的扩大卷积来增加感受野。扩大卷积（dilated convolution）是通过跳过部分输入来使filter可以应用于大于filter本身长度的区域。等同于通过增加零来从原始filter中生成更大的filter。

![](../../.gitbook/assets/1383870-20180730100217229-1516234421.gif)

