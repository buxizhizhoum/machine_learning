Environment:

    OS: ubuntu 16.04
    cpu: i5
    memory: 8G
    gpu: NV Geforce 920m
    python: python3.5
    tensorflow version: 1.5.0-dev20171127
    tensorflow-gpu version: 1.4.1

Scripts start with exercise is the exercises when learning tensorflow, and all
of them have been tested.

exercise开头的程序是学习时的练习代码，已经在上述环境中测试通过。

during the scripts above, the means of mostly used variables are listed below:

在上述的程序中，常用的一些变量的意义如下：
    
    x: input, in mnist, it is the images
    x: 输入
    y_: label of input, in mnist, it is the labels
    y_: 输出
    y: the prediction of model
    y: 预测值
    w: weight
    w: 权重
        
        there are other variables start with w, commonlly it is the weight of 
        different layer
       
    b: bias
    b: 偏置
        
        there are other variables start with b, commonlly it is the weight of 
        different layer
    
    train_x: train data sets input
    train_x: 训练数据
    train_y: train data label
    train_y: 训练数据的标签
    train_x_batch: batch of train data sets input
    train_x_batch: 一批训练数据
    train_y_batch: batch of train data label
    train_y_batch: 一批训练数据的标签
    test_x: test data sets input
    test_x: 测试数据
    test_y: test data label
    test_y: 测试数据的标签
    sess: session of tensorflow
    sess: 会话