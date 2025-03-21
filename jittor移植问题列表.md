## jittor移植问题列表

### 日期列表

[12月5日 S ](#D125)

[12月8日 S ](#D128)

[12月12日（讨论）](#D1212)

### <a name="D125">12月5日问题</a>

1. [该神经网络得到的输出应该是怎样的 S ](#Q1)
2. [jittor无法自动释放显存 S ](#Q2)

#### 1. <a name="Q1">该神经网络得到的输出应该是怎样的</a>

&emsp;&emsp;经过试训练和调试，我得到了初步的输出和记录，但发现该网络只得到了一些无意义的色块，同时网络的loss也没有减少。这应该是转换过的网络还需调试，但我想最好能找到一些标准的输出，这样也好寻找调试的方向。

&emsp;&emsp;暂时的解决方法是**把原项目用torch跑一遍**，但这样还要调试一遍环境，而且在之前的搜索中jittor和torch一起用好像会出bug。



#### Solved：

&emsp;&emsp;如果训练正常的，首先你的损失函数的值应该要持续下降，并且不是很高，统计的PSNR38以上，具体看论文介绍，然后中间过程得到的图片和log应该和训练集里真实的图片比较接近，export_mesh方法可以导出一个看起来比较正确的mesh，三角面片。

#### 2. <a name="Q2">jittor无法自动释放显存</a>

&emsp;&emsp;这是昨天（12.4）晚上训练时出现的问题，我设置了2万轮训练，这样的训练在我电脑上跑完是**正常**的，还可以再进行相同的训练，所以在完成之后我还想调一下参数重新进行训练，结果发现gpu的**显存**占满了，就连网络参数的梯度都没办法正常加载。显示文本如下：

```python
[e 1205 04:21:25.607197 36 executor.cc:682]
=== display_memory_info ===
 total_cpu_ram: 62.48GB total_device_ram: 10.75GB
 hold_vars: 481 lived_vars: 3606 lived_ops: 2996
 name: sfrl is_device: 1 used:  10.4GB(99.1%) unused: 96.31MB(0.896%) ULB:    40MB ULBO:    80MB total: 10.49GB
 name: sfrl is_device: 1 used:     0 B(-nan%) unused:     0 B(-nan%) total:     0 B
 name: sfrl is_device: 0 used:     0 B(-nan%) unused:     0 B(-nan%) total:     0 B
 name: sfrl is_device: 0 used: 184.5KB(18%) unused: 839.5KB(82%) total:     1MB
 name: sfrl is_device: 0 used:     0 B(-nan%) unused:     0 B(-nan%) total:     0 B
 name: temp is_device: 0 used:     0 B(-nan%) unused:     0 B(-nan%) total:     0 B
 name: temp is_device: 1 used:     0 B(-nan%) unused:     0 B(-nan%) total:     0 B
 cpu&gpu: 10.49GB gpu: 10.49GB cpu:     1MB
 free: cpu( 33.9GB) gpu(63.69MB)
 swap: total(    0 B) last(    0 B)
===========================

  0%|          | 1/20000 [00:00<3:33:42,  1.56it/s]
Traceback (most recent call last):
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/Runner.py", line 427, in <module>
    runner.train()
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/Runner.py", line 132, in train
    render_out = self.renderer.render(rays_o, rays_d, near, far,
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/models/renderer.py", line 428, in render
    ret_fine = self.render_core(rays_o,
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/models/renderer.py", line 234, in render_core
    inv_s = deviation_network(jittor.array(jittor.zeros([1, 3]))[:, :1].numpy().clip(1e-6, 1e6))  # Single parameter
RuntimeError: Wrong inputs arguments, Please refer to examples(help(jt.numpy)).

Types of your inputs are:
 self    = Var,
 args    = (),

The function declarations are:
 ArrayArgs fetch_sync()

Failed reason:[f 1205 04:21:25.607226 36 mem_info.cc:272]
*******************
GPU memory is overflow, please reduce your batch_size or data size!
Total: 10.75GB Used: 10.49GB
```

&emsp;&emsp;其中**`hold_vars: 481 lived_vars: 3606 lived_ops: 2996`**说明目前jittor占用的Var变量和Op算子是上一轮训练过程中保留的，结束进程之后他们还在，并且我并不知道该怎么清除这些已经占用的变量和算子。

&emsp;&emsp;在jittor官方的文档中写道添加如下代码可以强制回收内存：

```python
for ...:
    ...
    jt.sync_all()
    jt.gc()
```

&emsp;&emsp;我已添加但没有起到任何效果，于是我又添加了**`jt.display_memory_info()`**来显示内存，发现在模型加载完之后gpu就会占满，之后我又搜了很多释放gpu显存的教程，但没有jittor手动释放显存的方法，用torch的手动释放函数也无济于事。

>Jittor会输出内存消耗，以及计算图的大小**`lived_var,lived_op`**，以及用户持有的变量数`hold_var`, 如果计算图规模不断增大，请检查代码，或者提交github issue联系我们，并且附上错误日志和代码复现脚本。

&emsp;&emsp;jittor官方文档里是这么写的，但并没有给出任何解决方法和手动释放方法。现在的情况是**没办法进行训练测试**，也**没办法释放jittor的变量和算子**。另外我还加入了jittor的官方群，询问无果，但也有不少人遇到和这个一样的问题，有些人选择换显卡，但这样根本就没解决这个重复存储的问题，只是换了个新环境继续累计。



#### Solved：

&emsp;&emsp;更换更老的jittor版本，在1.3.5上下，这里换到了**1.3.5.1**。





### <a name="D128">12月8日问题</a>

1. [更换jittor版本后依旧无法释放内存](#Q1)

#### 1. <a name="Q1">更换jittor版本后依旧无法释放内存</a>

&emsp;&emsp;在得到回复之后我当天进行了版本的更换，结果还是得到了警告，并且gpu依旧被算子和变量占满，jittor的**1.3.5.1**版本选择的是gpu被占满之后**强行用cpu进行训练**，对于如此规模的训练而言，cpu训练的速度实在是太慢了。

&emsp;&emsp;于是我调小了训练规模，期望jittor能够在训练过程中自动清除无用的算子和变量，然而一天过去了，他并没有清除掉任何东西，依然是在训练过程中中断了，内存的调用依然是**gpu没有可用空间**，所以我暂时放弃了使用实验室电脑，选择一边查找解决办法一边用自己电脑进行小规模训练来调整网络参数。

&emsp;&emsp;目前还没查到任何的解决办法，已经去**官方的群和论坛**进行发帖询问了，暂时没得到回复，以下是错误报告：

```python
[w 1208 06:35:26.745127 08 grad.cc:77] grads[84] 'pts_fea.2.bias' doesn't have gradient. It will be set to zero: Var(2696:1:2:1:i0:o1:s1:n0,float32,pts_fea.2.bias,7fc6096f7400)[256,]
[w 1208 06:35:26.745130 08 grad.cc:77] grads[85] 'pts_fea.4.weight' doesn't have gradient. It will be set to zero: Var(2715:1:2:1:i0:o1:s1:n1,float32,pts_fea.4.weight,7fc6096f7800)[1,256,]

# 实际上这里有非常多条gardient消失的警告，这里只贴出来两条


[w 1208 06:35:34.426007 08 cuda_device_allocator.cc:29] Unable to alloc cuda device memory, use unify memory instead. This may cause low performance.
[i 1208 06:35:34.426015 08 cuda_device_allocator.cc:31]
=== display_memory_info ===
 total_cpu_ram: 62.48GB total_device_ram: 10.75GB
 hold_vars: 481 lived_vars: 5037 lived_ops: 5352
 name: sfrl is_device: 1 used: 8.829GB(98.1%) unused: 172.4MB(1.87%) total: 8.997GB
 name: sfrl is_device: 1 used:  1.33GB(99.7%) unused: 4.328MB(0.317%) total: 1.334GB
 name: sfrl is_device: 0 used:  1.33GB(99.7%) unused: 4.328MB(0.317%) total: 1.334GB
 name: sfrl is_device: 0 used: 523.5KB(51.1%) unused: 500.5KB(48.9%) total:     1MB
 name: temp is_device: 0 used:     0 B(-nan%) unused:     0 B(-nan%) total:     0 B
 name: temp is_device: 1 used:     0 B(0%) unused: 641.5KB(100%) total: 641.5KB
 cpu&gpu: 11.67GB gpu: 10.33GB cpu: 1.335GB
 free: cpu(14.12GB) gpu(23.69MB)
===========================

# 实际上这里每初始化一个网络，就会报一次内存信息，这里只贴出来一条
# 下面这一和之前没什么区别，只是总时间变长了

  0%|          | 1/20000 [00:00<1904:34:24,  326.23s/it]
Traceback (most recent call last):
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/Runner.py", line 427, in <module>
    runner.train()
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/Runner.py", line 132, in train
    render_out = self.renderer.render(rays_o, rays_d, near, far,
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/models/renderer.py", line 428, in render
    ret_fine = self.render_core(rays_o,
  File "/home/user/Desktop/Graduation-Work-main/JittorPaper1NERF/models/renderer.py", line 234, in render_core
    inv_s = deviation_network(jittor.array(jittor.zeros([1, 3]))[:, :1].numpy().clip(1e-6, 1e6))  # Single parameter
RuntimeError: Wrong inputs arguments, Please refer to examples(help(jt.numpy)).

Types of your inputs are:
 self    = Var,
 args    = (),

The function declarations are:
 ArrayArgs fetch_sync()

Failed reason:[f 1208 04:21:25.607226 36 mem_info.cc:272]
*******************
GPU memory is overflow, please reduce your batch_size or data size!
Total: 10.75GB Used: 10.49GB
```

&emsp;&emsp;这里的`lived_vars`等参数似乎**比之前多了**，这两天我先用我自己的电脑跑程序，等这个问题解决了再用实验室电脑跑。实际上我自己电脑跑程序是`2it/s`，实验室电脑是`3it/s`，也差不了多少，但这个问题确实不知道如何解决了。



#### Solved：

&emsp;&emsp;调试下来是电脑显存的问题，不是jittor的问题，将n_importance下调至32，降低gpu显存使用即可。



### <a name="D1212">12月12日讨论</a>

1. [调试过程中很多网络参数和结果输出为nan](#Q1)

#### 1. <a name="Q1">调试过程中很多网络参数和结果输出为nan</a>

&emsp;&emsp;在训练过程中，我注意到网络中的某些参数并非互相依赖的，即除了config文件中的那些参数之外还有一些独立存在的纯数字参数，更改gonfig文件之后其他的相应的也需要更改。

&emsp;&emsp;具体表现为：在我更改了n_importance之后，发现网络又跑不动了，具体调查原因是输入维数与网络要求的维数不同，于是我相应的修改了相关的参数之后又恢复了运行。

&emsp;&emsp;但正常运行的同时，某些不正常的运算也在进行，表现为数值矩阵经过神经网络处理之后变为nan矩阵，为此我决定将所有网络重新研究并重构一遍代码，来弄清究竟是哪里出了问题，但现在暂时还保持着原状，本意是今天征求一下学长意见再做下一步打算。

&emsp;&emsp;经过之前的资料查阅，我大约了解了nerf的网络结构，现在打算先了解数据集的结构和代码中其他网络的结构，之后再着手重构的工作。另外，在版本回退的过程中，我发现jittor版本过低会导致矩阵求逆函数的结果为nan和inf，而我将jittor版本升级到相对高（1.3.8）之后求逆函数会正常输出，而且参数降下来之后也没出现和之前一样的内存占用情况，所以我就先用相对高的jittor进行训练了。

