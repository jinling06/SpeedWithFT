

# FasterTransformer 加速流程
[知乎详细介绍](https://zhuanlan.zhihu.com/p/632763351)
## 配置环境
```python
# 下面配置在A6000上进行的实验 cuda11.4 pytorch=1.12.1+cu113
# 如果没有Git
apt-get install git
# 如果没有sudo
apt-get update
apt-get install sudo
# 配置cuda环境变量
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

# 更新cmake到最新版本 
apt remove cmake -y
sudo apt purge cmake
# cmake安装地址 https://cmake.org/files/v3.16/ 
# （最好安装最新版本，因为Git clone 的FT一般是最新版本，老版本容易报错）
wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2.tar.gz
tar zxvf cmake-3.25.2.tar.gz
cd cmake-3.25.2
sudo ./bootstrap  # 如果失败 执行 sudo apt-get install libssl-dev
sudo make  
sudo make install
cp /usr/local/bin/cmake /usr/bin/ (可选,3.25.2版本的不需要)
cmake  --version 
# 下载cudnn 放到cuda的目录下
# 下载地址: https://developer.nvidia.com/rdp/cudnn-archive (和cuda版本一致) 
# cuda11.4的话，里面包含 lib64 和 include 文件夹，放到/usr/local/cuda相应文件夹下
cudnn-11.4-linux-x64-v8.2.4.15



```
## 构建 FasterTransformer 动态库文件
```python
# 下载官方的 fastertransformer
git clone https://github.com/NVIDIA/FasterTransformer.git
# 定位到 文件夹
cd FasterTransformer/
git submodule init && git submodule update
# 创建build文件夹
mkdir -p build
cd build
# build pytorch版本的os文件 80是A100 详细看官方GitHub
# make 的时候报错 undefined reference to `MPI::Comm::Comm()' 则在cmake的时候加上 -DCMAKE_CXX_COMPILER=/usr/bin/mpic++
cmake -DCMAKE_CXX_COMPILER=/usr/bin/mpic++ -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
# 最后一步
make -j12 # 如果要清空之前的 需要 make clean
# 如果出现错误 
rm -rf libcudnn.so libcudnn.so.8
# 添加软连接
ln -s libcudnn.so.8.2.4 libcudnn.so.8
ln -s libcudnn.so.8.2.4 libcudnn.so
# 最终编译到 [98%] Built target gtest_main 也可测试下面的流程
# 最后一个bug是：fatal error: mpi.h: No such file or directory （未解决）
```

## 测试 bloomz-560m 模型效果

### HF测试
```python
# 安装依赖包
cd FasterTransformer
pip install -r examples/pytorch/gpt/requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
# 手动下载模型 放到 models
cd /workspace/model
https://huggingface.co/bigscience/bloomz-560m
# 下载数据集
cd /workspace/data
wget -c https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl 
# 转换数据格式
cd FasterTransformer
python examples/pytorch/gpt/utils/huggingface_bloom_convert.py --input-dir /root/xjl/workplace/models/bloomz-560m --output-dir /root/xjl/workplace/models/bloomz-560m-convert --data-type fp16 -tp 1 -v
# 转换模型
python examples/pytorch/gpt/utils/huggingface_bloom_convert.py --input-dir /root/xjl/workplace/models/bloomz-560m --output-dir /root/xjl/workplace/models/bloomz-560m-convert --data-type fp16
# 测试HF
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/gpt/bloom_lambada.py --tokenizer-path /root/xjl/workplace/models/bloomz-560m --dataset-path /root/xjl/workplace/data/lambada_test.jsonl --lib-path bulid/lib/libth_transformer.so --test-hf --show-progress

```
### FT 测试
```python
# 如果不能并行训练，则把 /root/xjl/FasterTransformer/examples/pytorch/gpt/utils/gpt.py 替换为
# 本项目中的 utils中的gpt.py
# 测试 FT
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/gpt/bloom_lambada.py --checkpoint-path /root/xjl/workplace/models/bloomz-560m-convert/1-gpu --tokenizer-path /root/xjl/workplace/models/bloomz-560m --dataset-path /root/xjl/workplace/data/lambada_test.jsonl --lib-path build/lib/libth_transformer.so --show-progress
```
### 结果
|模型名称|结果|
|---|---|
|huggingface|Accuracy: 39.4722% (2034/5153) (elapsed time: 112.0172 sec)|
|FasterTransformer|Accuracy: 39.4722% (2034/5153) (elapsed time: 11.1665 sec)|
|FasterTransformer with --data-type fp16|Accuracy: 39.4722% (2034/5153) (elapsed time: 10.8687 sec)|

## 测试 bart 加速

### 执行流程
```python
# 需要更换自己的model（model_name）和so文件地址（lib_path）
sh run_test.sh 
```
### 速度测试
#### 不同模型测试效果
参数：batch size=32, Input length=512, Output length=128, num_beams=3

|模型名称|HF结果|FT结果|提速|
|---|---|---|---|
|bart-base|1388.82 ms|129.39 ms|10.73|
|bart-base-chinese|1368.65 ms|128.35 ms|10.66|
|bart_finetuned_model_swa_base_ade_output|589.27 ms|71.63 ms|8.22|

#### 不同beam测试效果 
参数：model_name=bart-base, batch size=32, Input length=512, Output length=128 

|模型名称|HF结果|FT结果|提速|
|---|---|---|---|
|num_beams=1|527.27 ms|85.74 ms|6.15|
|num_beams=2|1088.06 ms|111.85 ms|9.73
|num_beams=3|1388.82 ms|129.39 ms|10.73|
|num_beams=4|1711.59 ms|148.36 ms|11.54|
|num_beams=5|2012.27 ms|187.18 ms|10.75|

#### 不同 batch_size 测试效果 
参数：model_name=bart-base, Input length=512, Output length=128, num_beams=3

|模型名称|HF结果|FT结果|提速|
|---|---|---|---|
|batch_size=1|369.30 ms|58.51 ms|6.31|
|batch_size=8|621.21 ms|75.64 ms|8.21|
|batch_size=16|860.32 ms|90.37 ms|9.52|
|batch_size=32|1388.82 ms|129.39 ms|10.73|
|batch_size=64|2532.13 ms|215.92 ms|11.73|
|batch_size=128|5053.91 ms|391.86 ms|12.90|

注：\
1、[bart-base](https://huggingface.co/facebook/bart-base) \
2、[bart-base](https://huggingface.co/fnlp/bart-base-chinese) \
3、bart_finetuned_model_swa_base_ade_output：bart-base-chinese缩小词表（大小1734）微调的模型 \
4、上述结果均是10轮平均时间 \

### 说明
1、FT的结果和HF差不多一致，有时候不会完全一致，但相差无几 \
2、FT符合正常预期，beam越大，加速倍数越多，batch_size越大，加速倍数越多。
(之前实现的一个onnx是batch越大，加速越少，很奇怪)
 
## 测试bert（2分类） 加速
### 执行流程
```python
# 取消注释 python bert_test.py  相关部分
sh run_test.sh
```
### 速度测试
#### 有无 remove_padding
参数：[bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main) ,
batch_size=32, use_fp16=True,input_max_len=512

|模型名称|HF结果|FT结果|提速|
|---|---|---|---|
|remove_padding True|92.40 ms|7.13 ms|12.96
|remove_padding False|88.73 ms|7.16 ms|12.39

#### 不同 batch_size 速度测试
参数：[bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main) ,
remove_padding=False, use_fp16=True, input_max_len=512

|模型名称|HF结果|FT结果|提速|
|---|---|---|---|
|batch_size=1|7.94 ms|1.75 ms|4.54|
|batch_size=8|35.26 ms|2.27 ms|15.52|
|batch_size=16|59.13 ms|3.95 ms|14.95|
|batch_size=32|88.73 ms|7.16 ms|12.39
|batch_size=64|163.18 ms|13.87 ms|11.77|

### 参考资料
[1] [大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650437225&idx=1&sn=ccae9c06d4e34b5252fa88c39232df3f&chksm=becdf03389ba792574363a8f558f87b0c284792e8703a155d139403970237d3cbb6c6920afa0&mpshare=1&scene=23&srcid=0519Xt8wE1C4Sy1q9kq49CCg&sharer_sharetime=1684508767910&sharer_shareid=c75e5c5178583e00951be2bee1d649c0#rd)   \
[2] https://github.com/NVIDIA/FasterTransformer/tree/main
