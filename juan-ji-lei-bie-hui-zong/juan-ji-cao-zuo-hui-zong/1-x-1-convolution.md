# 1 x 1 Convolution

## 概述

如果输入层具有多个通道，使用1\*1卷积将十分有趣。下图说明了如何对尺寸为H xW x D的输入层进行1 x 1卷积。在对尺寸为1 x 1 x D的filter进行1 x 1卷积之后，输出通道的尺寸为H x W x 1。应用N个这样的1 x 1卷积，然后将结果串联在一起，我们可以得到尺寸为H xW x N的输出层。

![](../../.gitbook/assets/image%20%28116%29.png)

最初，在[论文](https://arxiv.org/pdf/1312.4400.pdf)中提出了1 x 1卷积。然后，它们在Google Inception论文中得到了广泛使用。1 x 1卷积的一些优点是：

* 可以实现升降维
* 降维以实现高效计算
* 高效的低维嵌入或特征池化
* 卷积后再次应用非线性操作，以便与网络学习更复杂的功能

在上图中可以看到前两个优点。经过1 x 1卷积后，我们在深度方向上显着减小了尺寸。假设原始输入有200个通道，则1 x 1卷积会将这些通道（功能）嵌入到单个通道中。第三个优点是在1 x 1卷积之后，可以添加非线性激活，例如ReLU。非线性允许网络学习更复杂的功能。

这些优势在Google的Inception[论文](https://arxiv.org/pdf/1409.4842.pdf)中描述为：

> “One big problem with the above modules, at least in this naïve form, is that even a modest number of 5x5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters. 
>
> This leads to the second idea of the proposed architecture: judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise. This is based on the success of embeddings: even low dimensional embeddings might contain a lot of information about a relatively large image patch… That is, 1 x 1 convolutions are used to compute reductions before the expensive 3 x 3 and 5 x 5 convolutions. Besides being used as reductions, they also include the use of rectified linear activation which makes them dual-purpose.”

关于1 x 1卷积的一个有趣观点来自Yann LeCun，“在卷积网中，没有“全连接层”之类的东西。只有带有1x1卷积内核和完整连接表的卷积层。”

## 1x1卷积

在GoogLeNet的Inception模块中，有1x1的卷积核。这初看起来是一个非常奇怪的做法，因为1x1的卷积核基本上失去了卷积的作用，并没有建立在同一个通道上的相邻像素之间的相关性。

在本例中，为了识别颜色，我们也使用了1x1的卷积核，并且能够完成颜色分类的任务，这是为什么呢？

我们以三通道的数据举例。

![&#x56FE;18-19 1x1&#x5377;&#x79EF;&#x6838;&#x7684;&#x5DE5;&#x4F5C;&#x539F;&#x7406;](../../.gitbook/assets/image%20%2862%29.png)

假设有一个三通道的1x1的卷积核，其值为\[1,2,-1\]，则相当于把每个通道的同一位置的像素值乘以卷积核，然后把结果相加，作为输出通道的同一位置的像素值。以左上角的像素点为例：

$$
1 \times 1 + 1 \times 2 + 1 \times (-1)=2
$$

相当于把上图拆开成9个样本，其值为：

```python
[1,1,1] # 左上角点
[3,3,0] # 中上点
[0,0,0] # 右上角点
[2,0,0] # 左中点
[0,1,1] # 中点
[4,2,1] # 右中点
[1,1,1] # 左下角点
[2,1,1] # 下中点
[0,0,0] # 右下角点
```

上述值排成一个9行3列的矩阵，然后与一个3行1列的向量$$(1,2,-1)^T$$相乘，得到9行1列的向量，然后再转换成3x3的矩阵。当然在实际过程中，这个1x1的卷积核的数值是学习出来的，而不是人为指定的。

这样做可以达到两个目的：

1. 跨通道信息整合
2. 降维以减少学习参数

**所以1x1的卷积核关注的是不同通道的相同位置的像素之间的相关性**，而不是同一通道内的像素的相关性，在本例中，意味着它关心的彩色通道信息，通过不同的卷积核，把彩色通道信息转变成另外一种表达方式，在保留原始信息的同时，还实现了降维。

在本例中，第一层卷积如果使用3个卷积核，输出尺寸是3x28x28，和输入尺寸一样，达不到降维的作用。所以，一般情况下，会使用小于输入通道数的卷积核数量，比如输入通道为3，则使用2个或1个卷积核。在上例中，如果使用2个卷积核，则输出两张9x9的特征图，这样才能达到降维的目的。如果想升维，那么使用4个以上的卷积核就可以了。

## 总结

单通道使用1x1卷积不会实现升降维，但是对于多通道来说，使用1x1卷积会降维。

对于256x3x3操作，我们可以使用1x1降维到64，然后用64x256x3x3，最后通过1x1再进行升维，通过这样操作可以降低约10倍左右的计算量。

