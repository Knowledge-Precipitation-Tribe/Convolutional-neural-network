# Pointwise Grouped Convolution

ShuffleNet论文还介绍了逐点分组卷积。通常，对于诸如MobileNet或ResNeXt中的分组卷积，分组操作是在3x3空间卷积上执行的，而不是在1x1卷积上执行的。

shuffleNet论文认为1 x 1卷积在计算上也很昂贵。它建议将组卷积也应用于1 x 1卷积。顾名思义，逐点分组卷积执行1 x 1卷积的分组操作。该操作与分组卷积的操作相同，只是进行了一种修改-对1x1滤镜而不是NxN滤镜（N&gt; 1）执行。

在ShuffleNet论文中，作者利用了我们学到的三种类型的卷积：（1）分组卷积；（2）逐点分组卷积；（3）深度可分离卷积。这样的架构设计在保持精度的同时大大降低了计算成本。例如，ShuffleNet和AlexNet的分类错误在实际的移动设备上是可比的。但是，计算成本已从AlexNet中的720 MFLOP大幅降低到ShuffleNet中的40–140 MFLOP。ShuffleNet具有相对较低的计算成本和良好的模型性能，在用于移动设备的卷积神经网络领域中很受欢迎。

