# 通过无监督算法来分析prads日志的行为

## 参考
- [社区流量差异分析](https://d3fend.mitre.org/technique/d3f:NetworkTrafficCommunityDeviation)
- [工作职能访问模式分析](https://d3fend.mitre.org/technique/d3f:JobFunctionAccessPatternAnalysis)

## 项目说明
- 代码环境


```

1. 将数据提取为合适的特征，
    - 通过特征工程提取，例如提取时间的属性，小时，周，工作日等
    - 提取网络区域的特征，`vlan,distance` 归纳为网络空间的位置，是否容易被不合规
    - 将访问源的资产属性进行聚类为去冗余信息
2. 使用自编码的算法进行训练，线不考虑 `ret`; 将所有的数据作为使用，最后对无监督的模型进行确认。

```