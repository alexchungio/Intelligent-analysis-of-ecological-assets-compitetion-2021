# Intelligent-analysis-of-ecological-assets-compitetion-2021
[天池2021全国数字生态创新大赛 智能算法赛：生态资产智能分析](https://tianchi.aliyun.com/competition/entrance/531860/introduction?lang=en-us)


## EDA
[data analysis](./docs/data_analysis.ipynb)

## requirements
* segmentation-models-pytorch
```shell script
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

## Train
```shell script
python train.py
```

## inference
```shell script
python inference.py
```

## submit
```shell script
zip -r results.zip results
```

## Logs
### 2020-02-04
* 修改代码中bug(test 中图像格式需要从BGR 转换为 RGB)
* 线上分数 0.3406

### 2020-02-05
* 增加 re-weight 策略
* 线上分数 0.3511 （线下验证集 miou 分数下降， 线上分数反而提高了， 可能是因为训练集和测试的数据分布不同导致的）
   
### 2020-02-06
* **更换baseline**
* backbone: efficient
* model: unet++
* image_scale: 256
* 联合loss: dice_loss + label_smooth
* optimize: adamW
* scheduler: warmUpConsineScheduler
* batch_size: 6
* num_train:num_val = 0.85:0.15
* lr: 3e-4
* 线上分数 0.3712
   
### 2020-02-09
* epoch: 44
* **增大epoch: 30 -> 44**
* 线上分数 0.3763
 
### 2020-02-18
* epoch 44
* class_weights: [0.25430845, 0.24128766, 0.72660323, 0.57558217, 0.74196072, 0.56340895, 0.76608468, 0.80792181, 0.47695224, 1.]
* **增加 re-weight 策略**
  不 wrok
  
### 2020-02-22
* num_train:num_val = **1 : 0.15**
* batch_size: 6
* epoch 44
* **使用全量数据进行训练**
* 线上分数 0.3768 
 
### 2020-02-24
* num_train:num_val = 1 :0.15
* batch_size: 6 * 3 = 18
* epoch 44
* **增大 batch-size: 6 -> 18**

  使用 gradient accumulate 训练, 进行梯度累加要注意batch_size 与 learning_rate 匹配的问题：
  * **通常来说，batch_size 与 learning rate 要等比例地放大**, 此时不需要修改loss, **累计多个batch的loss之和， 再执行一次更新**， 本质上相当于扩大了learning rate，代码如下 
    ```python
    # 2.1 loss regularization
    regular_loss = loss
    # 2.2 back propagation
    scaler.scale(regular_loss).backward()
    # 2.3 update parameters of net
    if (batch_idx + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step(epoch + batch_idx / train_loader_size)
    ```
  * 如果由于GPU显存的限制，只希望扩大batch_size, 同时保持学习大小不变，此时需要**对累计的多个batch的loss求平均， 再执行一次更新**， 代码如下
    ```python
    # 2.1 loss regularization
    regular_loss = loss / accumulation_steps
    # 2.2 back propagation
    scaler.scale(regular_loss).backward()
    # 2.3 update parameters of net
    if (batch_idx + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step(epoch + batch_idx / train_loader_size)
    ```
  如果 batch size 大了，那么 gradient 里的 noise 会变小，SGD 越来越像 gradient descent， 此时迭代时的泛化性可能会降低。
  
* 线上分数 0.3808

### 2020-02-25
* num_train:num_val = 1 :0.15
* batch_size: 16
* epoch 44
* 线上分数 0.3820

### 2020-02-26
* num_train:num_val = 1 :0.15
* batch_size: 16
* optimize: SGDM
* scheduler: NoamLR
* lr: 0.01
* epoch 80
 
### trick
* warmup
* re-weight
* re-sample
* dice loss
* lovasz_loss
* CosineAnnealingWarmRestarts
* data augmentation
* TTA
* 针对 grass、construction、 bareland 训练二分类模型
* SWA(Stochastic Weights Averaging)

## TODO

## Reference

* <https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.6cc26423XSMkYs&postId=169396>
* <https://github.com/MrGiovanni/UNetPlusPlus>
* <https://github.com/DLLXW/data-science-competition>
* <https://github.com/tugstugi/pytorch-saltnet>
* <https://www.zhihu.com/question/375988016/answer/1698592088>