# Intelligent-analysis-of-ecological-assets-compitetion-2021
[天池2021全国数字生态创新大赛 智能算法赛：生态资产智能分析](https://tianchi.aliyun.com/competition/entrance/531860/introduction?lang=en-us)


## EDA
[data analysis](./docs/data_analysis.ipynb)

## Train

## inference

## submit

```shell script
zip -r results.zip results
```

## Logs

### 2020-02-05

* 修改代码中bug(test 中图像格式需要从BGR 转换为 RGB)
  线上测试 0.3406

* 增加 re-weight 策略
  线上分数 0.3511 （线下验证集 miou 分数下降， 线上分数反而提高了， 可能是因为训练集和测试的数据分布不同导致的）
   


## Reference

* <https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.6cc26423XSMkYs&postId=169396>
* <https://github.com/MrGiovanni/UNetPlusPlus>
* <https://aistudio.baidu.com/aistudio/projectdetail/1465819?spm=5176.12282029.0.0.56951df1MUxsWK>