2020.08.26 —— 2020.08.28
1. allen 文件夹是尝试使用 AllenNLP 框架的代码，但是后面发现其文档还未更新完全，所以先放弃了这一版 😅
2. 目前主要使用 huggingface 文件夹中的代码，在这个基础上进行改进
- 目前 jaccard 得分为 0.69254



2020.08.31 —— 2020.09.05

![实验记录](https://i.loli.net/2020/09/06/Pk6X7N5CobRKSh8.png)
* 表中 \* 表示一致
* last2h 表示使用最后两层隐藏状态

实验小结：  
1. 小的 batch size 会有更好的表现，64 或者 128
2. task layer 添加 dropout 和 特殊初始化没什么作用
3. 对于 sentiment 为 neutral 的样本直接使用原文作为答案可以提升效果
4. 模型上 bert-wwm 类最佳，其次 bert，roberta 表现最差，很奇怪，和讨论区的观点不太一致。
另外，经过 squad 数据 fine-tune 的模型并没有预想的表现的比未经过 fine-tune 的好




2020.09.07 —— 2020.09.


|   model   |   pooler   |   hidden-state   |   dropout   |   fc-normal-init   |   epoch   |   lr   |   bs   |   max-seq-len   |   fp16/o1   |   weight-decay   |   best-steps/total-steps   |   n-gpus   |   warmup-steps   |   jaccard (best/final)   |   CELoss   |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   albert-large-v2   |   0   |   last2h   |   0   |   0   |   5   | 2e-5 | 32 | 128 | 1/o1 | 0 | 800/880 | 4 | 0 | 70.1796/70.0795 | 0.5939 |
| albert-large-v2 | 0 | last2h | 0 | 0 | 5 | 3e-5 | 32 | 128 | 1/o1 | 0 | 750/880 | 4 | 0 | 69.5190/69.5166 | 0.5375 |
| albert-large-v2 | 0 | Last2h | 0.1 | 0 | 5 | 2e-5 | 32 | 128 | 1/o1 | 0 | 850/880 | 4 | 0 | 69.9607/69.9118 | 0.6085 |
| albert-large-v2 | 0 | Last2h | 0.1 | 0 | 5 | 3e-5 | 32 | 128 | 1/o1 | 0 | 800/880 | 4 | 0 | 69.8822/69.8574 | 0.5515 |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |


