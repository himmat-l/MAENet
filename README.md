# MAENet
It's a project for multi-task NN with semantic segmentation and object detection task

#Notice
1.PIL和pytorch的图像resize操作，与opencv的resize结果不一样，这样会导致训练采用PIL，预测时采用opencv，结果差别很大，尤其是在检测和分割任务中比较明显。
2.PyTorch中在反向传播前为什么要手动将梯度清零？
1)从内存消耗的角度来看。这种模式可以让梯度玩出更多花样，
  比如说上面讲到的梯度累加（gradient accumulation）实现的“显存受限”解决：
  在内存大小不够的情况下叠加多个batch的grad作为一个大batch进行迭代，因为二者得到的梯度是等价的,即当你GPU显存较少时，你又想要调大batch-size，此时你就可以利用PyTorch的这个性质进行梯度的累加来进行backward
2)利用梯度累加，可以在最多保存一张计算图的情况下进行multi-task任务的训练：
  从PyTorch的设计原理上来说，在每次进行前向计算得到pred时，会产生一个用于梯度回传的计算图，这张图储存了进行back propagation需要的中间结果，当调用了.backward()后，会从内存中将这张图进行释放。
3.torch.cuda.empty_cache()
Pytorch 训练时无用的临时变量可能会越来越多，导致 out of memory ，可以使用下面语句来清理这些不需要的变量：torch.cuda.empty_cache() 。
4.训练时冻结一部分参数
  在中间插入代码，这样前面的参数就是False，而后面的不变
  self.conv1 = nn.Conv2d(1, 6, 5)
  self.conv2 = nn.Conv2d(6, 16, 5)
  #将要冻结的参数的requires_grad设为False
  for p in self.parameters():
  p.requires_grad=False
  self.fc1 = nn.Linear(16 * 5 * 5, 120)
  self.fc2 = nn.Linear(120, 84)
  self.fc3 = nn.Linear(84, 10)