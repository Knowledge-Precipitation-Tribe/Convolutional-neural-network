# 池化的前向计算与反向传播

## 常用池化方法

池化 pooling，又称为下采样，downstream sampling or sub-sampling。

池化方法分为两种，一种是最大值池化 Max Pooling，一种是平均值池化 Mean/Average Pooling。如图17-32所示。

![&#x56FE;17-32 &#x6C60;&#x5316;](../.gitbook/assets/image%20%2814%29.png)

* 最大值池化，是取当前池化视野中所有元素的最大值，输出到下一层特征图中。
* 平均值池化，是取当前池化视野中所有元素的平均值，输出到下一层特征图中。

其目的是：

* 扩大视野：就如同先从近处看一张图片，然后离远一些再看同一张图片，有些细节就会被忽略
* 降维：在保留图片局部特征的前提下，使得图片更小，更易于计算
* 平移不变性，轻微扰动不会影响输出：比如上图中最大值池化的4，即使向右偏一个像素，其输出值仍为4
* 维持同尺寸图片，便于后端处理：假设输入的图片不是一样大小的，就需要用池化来转换成同尺寸图片

一般我们都使用最大值池化。

## 池化的其它方式

在上面的例子中，我们使用了size=2x2，stride=2的模式，这是常用的模式，即步长与池化尺寸相同。

我们很少使用步长值与池化尺寸不同的配置，所以只是提一下，如图17-33。

![&#x56FE;17-33 &#x6B65;&#x957F;&#x4E3A;1&#x7684;&#x6C60;&#x5316;](../.gitbook/assets/image%20%285%29.png)

上图是stride=1, size=2x2的情况，可以看到，右侧的结果中，有一大堆的3和4，基本分不开了，所以其池化效果并不好。

假设输入图片的形状是 $$W_1 \times H_1 \times D_1$$，其中W是图片宽度，H是图片高度，D是图片深度（多个图层），F是池化的视野（正方形），S是池化的步长，则输出图片的形状是：

$$
\begin{aligned} W_2 &= (W_1 - F)/S + 1 \ H_2 = (H_1 - F)/S + 1 \ D_2 &= D_1 \end{aligned}
$$

池化层不会改变图片的深度，即D值前后相同。

## 池化层的训练

我们假设图17-34中，$$[[1,2],[3,4]]$$是上一层网络回传的残差，那么：

* **对于最大值池化，残差值会回传到当初最大值的位置上，而其它三个位置的残差都是0。**
* 对于平均值池化，残差值会平均到原始的4个位置上。

![&#x56FE;17-34 &#x5E73;&#x5747;&#x6C60;&#x5316;&#x4E0E;&#x6700;&#x5927;&#x6C60;&#x5316;](../.gitbook/assets/image%20%2858%29.png)

![&#x56FE;17-35 &#x6C60;&#x5316;&#x5C42;&#x53CD;&#x5411;&#x4F20;&#x64AD;&#x7684;&#x793A;&#x4F8B;](../.gitbook/assets/image%20%288%29.png)

### Max Pooling

严格的数学推导过程以图17-35为例进行。

正向公式：

$$ w = max(a,b,e,f) $$

反向公式（假设Input Layer中的最大值是b）：

$$ {\partial w \over \partial a} = 0, \quad {\partial w \over \partial b} = 1 $$

$$ {\partial w \over \partial e} = 0, \quad {\partial w \over \partial f} = 0 $$

因为a,e,f对w都没有贡献，所以偏导数为0，只有b有贡献，偏导数为1。

$$ \delta_a = {\partial J \over \partial a} = {\partial J \over \partial w} {\partial w \over \partial a} = 0 $$

$$ \delta_b = {\partial J \over \partial b} = {\partial J \over \partial w} {\partial w \over \partial b} = \delta_w \cdot 1 = \delta_w $$

$$ \delta_e = {\partial J \over \partial e} = {\partial J \over \partial w} {\partial w \over \partial e} = 0 $$

$$ \delta_f = {\partial J \over \partial f} = {\partial J \over \partial w} {\partial w \over \partial f} = 0 $$

### Mean Pooling

正向公式：

$$w = \frac{1}{4}(a+b+e+f)$$

反向公式（假设Layer-1中的最大值是b）：

$$ {\partial w \over \partial a} = \frac{1}{4}, \quad {\partial w \over \partial b} = \frac{1}{4} $$

$$ {\partial w \over \partial e} = \frac{1}{4}, \quad {\partial w \over \partial f} = \frac{1}{4} $$

因为a,b,e,f对w都有贡献，所以偏导数都为1：

$$ \delta_a = {\partial J \over \partial a} = {\partial J \over \partial w} {\partial w \over \partial a} = \frac{1}{4}\delta_w $$

$$ \delta_b = {\partial J \over \partial b} = {\partial J \over \partial w} {\partial w \over \partial b} = \frac{1}{4}\delta_w $$

$$ \delta_e = {\partial J \over \partial e} = {\partial J \over \partial w} {\partial w \over \partial e} = \frac{1}{4}\delta_w $$

$$ \delta_f = {\partial J \over \partial f} = {\partial J \over \partial w} {\partial w \over \partial f} = \frac{1}{4}\delta_w $$

无论是max pooling还是mean pooling，都没有要学习的参数，所以，在卷积网络的训练中，池化层需要做的只是把误差项向后传递，不需要计算任何梯度。

## 实现方法1

按照标准公式来实现池化的正向和反向代码。

```python
class PoolingLayer(CLayer):
    def forward_numba(self, x, train=True):
        ......

    def backward_numba(self, delta_in, layer_idx):
        ......
```

有了前面的经验，这次我们直接把前向和反向函数用numba方式来实现，并在前面加上@nb.jit修饰符：

```python
@nb.jit(nopython=True)
def jit_maxpool_forward(...):
    ...
    return z

@nb.jit(nopython=True)
def jit_maxpool_backward(...):
    ...
    return delta_out
```

## 实现方法2

池化也有类似与卷积优化的方法来计算，在图17-36中，我们假设大写字母为池子中的最大元素，并且用max\_pool方式。

![&#x56FE;17-36 &#x6C60;&#x5316;&#x5C42;&#x7684;img2col&#x5B9E;&#x73B0;](../.gitbook/assets/image%20%2816%29.png)

原始数据先做img2col变换，然后做一次np.max\(axis=1\)的max计算，会大大增加速度，然后把结果reshape成正确的矩阵即可。做一次大矩阵的max计算，比做4次小矩阵计算要快很多。

```python
class PoolingLayer(CLayer):
    def forward_img2col(self, x, train=True):
        ......

    def backward_col2img(self, delta_in, layer_idx):
        ......
```

## 性能测试

下面我们要比较一下以上两种实现方式的性能，来最终决定使用哪一种。

对同样的一批64个样本，分别用两种方法做5000次的前向和反向计算，得到的结果：

```python
Elapsed of numba: 17.537396907806396
Elapsed of img2col: 22.51519775390625
forward: True
backward: True
```

numba方法用了17秒，img2col方法用了22秒。并且两种方法的返回矩阵值是一样的，说明代码实现正确。

## 代码位置

原代码位置：[ch17, Level5](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch17-CNNBasic/Level5_PoolLayer_Test.py)

个人代码：[**PoolLayer\_Test**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/PoolLayer_Test.py)\*\*\*\*
