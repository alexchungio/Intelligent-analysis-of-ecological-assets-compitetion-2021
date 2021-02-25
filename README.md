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

## inference

## submit

```shell script
zip -r results.zip results
```

## Logs

### 2020-02-05

* 修改代码中bug(test 中图像格式需要从BGR 转换为 RGB)
  线上分数 0.3406

* 增加 re-weight 策略
  线上分数 0.3511 （线下验证集 miou 分数下降， 线上分数反而提高了， 可能是因为训练集和测试的数据分布不同导致的）
   
### 2020-02-06
* 更换baseline  efficient+unet++
  线上分数 0.3712
  
  
### 2020-02-09
* 增大epoch
  线上分数 0.3768
   
### 2020-02-18
* 增加 re-weight 策略
  不 wrok
  

### 2020-02-20
* 增大batch-size
  使用 gradient accumulate 训练, 克服GPU显存限制
  batch_size: 6 -> 18
  线上分数 0.3808
  
  
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