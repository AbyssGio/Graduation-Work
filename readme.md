##1.已知问题：
jittor框架在进行并行矩阵相乘运算时使用GPU加速会检测矩阵的前n维，导致报错，但CPU模式不会出问题。

###目前解决办法
在本地jittor库里修改代码： `jittor/nn.py`文件中注释`120~124`行，让矩阵相乘不进行GPU加速。

##2.运行此项目：
###运行JittorPaper1NeRF-HSR
1. 修改`JittorPaper1NeRF-HSR/confs/womask.conf`文件，将数据集和存储路径改为本地路径。
2. 在`JittorPaper1NeRF-HSR/exp_runner.py`文件`365`行之后修改参数默认值。
3. 运行`JittorPaper1NeRF-HSR/exp_runner.py`文件，等待实验输出。

###运行JittorPaper2CDF
1. 修改`JittorPaper2CDF/experiments/df_L50.yml`文件，将数据集和存储路径改为本地路径。
2. 运行`JittorPaper2CDF/main.py`文件，等待实验输出。

##3.数据集下载：
1. 通过百度网盘连接`https://pan.baidu.com/s/1ED0qKmvSr2h5Il15IJ50jA?pwd=1234` 下载。
2. 解压`dataset.zip`文件，其中`dataset1`文件夹是`JittorPaper1NeRF-HSR`的数据集，`dataset2`文件夹是`JittorPaper2CDF`的数据集。
