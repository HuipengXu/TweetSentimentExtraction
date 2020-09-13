**2020.08.26 —— 2020.08.28**

1. allen 文件夹是尝试使用 AllenNLP 框架的代码，但是后面发现其文档还未更新完全，所以先放弃了这一版 😅
2. 目前主要使用 huggingface 文件夹中的代码，在这个基础上进行改进
- 目前 jaccard 得分为 0.69254



**2020.08.31 —— 2020.09.05**

![实验记录](https://i.loli.net/2020/09/06/Pk6X7N5CobRKSh8.png)
* 表中 \* 表示一致
* last2h 表示使用最后两层隐藏状态

实验小结：  
1. 小的 batch size 会有更好的表现，64 或者 128
2. task layer 添加 dropout 和 特殊初始化没什么作用
3. 对于 sentiment 为 neutral 的样本直接使用原文作为答案可以提升效果
4. 模型上 bert-wwm 类最佳，其次 bert，roberta 表现最差，很奇怪，和讨论区的观点不太一致。
另外，经过 squad 数据 fine-tune 的模型并没有预想的表现的比未经过 fine-tune 的好



**2020.09.07 —— 2020.09.11**

* 1st Models


|   1st-model   |   jacquard-based soft labels   |   MSD   |   hidden-state   |   dropout   |   fc-normal-init   |   epoch   |   lr   |   bs   |   max-seq-len   |   fp16/o1   |   weight-decay   |   best-steps/total-steps   |   n-gpus   |   warmup-steps   |   jaccard (best/final)   |   CELoss   |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   albert-large-v2   |   0   |      |   last2h   |   0   |   0   |   5   | 2e-5 | 32 | 128 | 1/o1 | 0 | 800/880 | 4 | 0 | 70.1796/70.0795 | 0.5939 |
| albert-large-v2 | 0 |  | last2h | 0 | 0 | 5 | 3e-5 | 32 | 128 | 1/o1 | 0 | 750/880 | 4 | 0 | 69.5190/69.5166 | 0.5375 |
| albert-large-v2 | 0 |  | last2h | 0.1 | 0 | 5 | 2e-5 | 32 | 128 | 1/o1 | 0 | 850/880 | 4 | 0 | 69.9607/69.9118 | 0.6085 |
| albert-large-v2 | 0 |  | last2h | 0.1 | 0 | 5 | 3e-5 | 32 | 128 | 1/o1 | 0 | 800/880 | 4 | 0 | 69.8822/69.8574 | 0.5515 |
| albert-large-v2 | 0 |  | last2h | 0 | Normal(std=0.02) | 5 | 2e-5 | 32 | 128 | 1/o1 | 0 | 750/880 | 4 | 0 | 70.0787/69.9847 | 0.6002 |
| albert-large-v2 | 0 |  | last2h | 0 | 0 | 4 | 2e-5 | 32 | 128 | 1/o1 | 0 | 600/704 | 4 | 0 | **70.2961**/70.2722 | 0.6488 |
| albert-large-v2 | 0 |  | last2h | 0 | 0 | 3 | 2e-5 | 32 | 128 | 1/o1 | 0 | 450/520 | 4 | 0 | 70.0191/69.8680 | 0.7186 |
| albert-large-v2 | 0 |  | cat(avg-all-hidden,max-all-hidden) | 0 | 0 | 3 | 2e-5 | 32 | 128 | 1/o1 | 0 | */528 | 4 | 0 | 69.6336/* | 0.7524 |
| albert-large-v2 | 0 |  | cat(avg-all-hidden,max-all-hidden) | 0.1 | 0 | 3 | 2e-5 | 32 | 128 | 1/o1 | 0 | */528 | 4 | 0 | 69.5679/* | 0.7548 |
| albert-large-v2 | 0 |  | linear-comb-hidden | 0 | 0 | 3 | 2e-5 | 32 | 128 | 1/o1 | 0 | 450/528 | 4 | 0 | 69.2622/68.9154 | 0.7793 |
| albert-large-v2 | 0 |  | linear-comb-hidden | 0 | 0 | 4 | 2e-5 | 32 | 128 | 1/o1 | 0 | */704 | 4 | 0 | 69.3700/* | 0.6970 |
| albert-large-v2 | 0 | 1 | last2h | 0 | 0 | 4 | 2e-5 | 32 | 128 | 1/o1 | 0 | 650/704 | 4 | 0 | 69.9925/69.8471 | 0.6974 |
| albert-large-v2 | 0 | 1 | last2h | 0 | 0 | 5 | 2e-5 | 32 | 128 | 1/o1 | 0 | 750/880 | 4 | 0 | 70.0275/69.8508 | 0.6432 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Roberta-base | 0 |  | last2h | 0 | 0 | 4 | 2e-5 | 32 | 128 | 0 | 0 | 650/704 | 4 | 0 | 69.2620/69.1224 | 0.7728 |
| Roberta-base | 0 |  | last2h | 0 | 0 | 4 | 5e-5 | 32 | 128 | 0 | 0 | */704 | 4 | 0 | 69.7026/* | 0.6525 |
| Roberta-base | 0 |  | last2h | 0 | 0 | 4 | 5e-5 | 64 | 128 | 0 | 0 | 300/352 | 4 | 0 | 69.8618/69.7472 | 0.7231 |
| Roberta-base | 0 |  | last2h | 0 | 0 | 5 | 5e-5 | 64 | 128 | 0 | 0 | 400/440 | 4 | 0 | 69.8478/69.8340 | 0.6757 |
| Roberta-base | 0 |  | last2h | 0 | 0 | 5 | 5e-5 | 64 | 128 | 0 | 0 | */590 | 3 | 0 | **70.0300**/* | 0.6437 |
| Roberta-base | 0 | 1 | last2h | 0 | 0 | 5 | 5e-5 | 64 | 128 | 0 | 0 | */590 | 3 | 0 | 69.6127/* | 0.6656 |
| Roberta-base | 0 | 1 | last2h | 0 | 0 | 6 | 5e-5 | 64 | 128 | 0 | 0 | */708 | 3 | 0 | 69.7245/* | 0.5994 |
| Roberta-base | 0 | 1 | last2h | 0 | 0 | 6 | 5e-5 | 128 | 128 | 1/o1 | 0 | */354 | 3 | 0 | 69.7075/* | 0.6964 |
| Roberta-base | 1 | 0 | last2h | 0 | 0 | 5 | 5e-5 | 64 | 128 | 0 | 0 | */590 | 3 | 0 | 77.33/* | 过度预处理 selelcted _text 了 |
| Roberta-base | 1 | 0 | last2h | 0 | 0 | 5 | 5e-5 | 64 | 128 | 0 | 0 | */590 | 3 | 0 | **71.7002**/* | 1.65e-3 (KL) |
| Roberta-base | 1 | 0 | last2h | 0 | 0 | 6 | 5e-5 | 64 | 128 | 0 | 0 | 700/708 | 3 | 0 | 71.6937/71.6521 | 1.48e-3 (KL) |
| Roberta-base | 1 | 0 | last2h+conv | 0 | 0 | 6 | 5e-5 | 64 | 128 | 0 | 0 | /708 | 3 | 0 | **71.7433**/71.6613 | 1.46e-3 (KL) |
| Roberta-base | 0 | 0 | last2h+conv | 0 | 0 | 5 | 5e-5 | 64 | 128 | 0 | 0 | 550/590 | 3 | 0 | 71.4312/71.3696 | 0.6223 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| bert-base-cased | 0 | 1 | cat all hidden (12 layers) | 0 | 0 | 5 | 2e-5 | 32 | 128 | 0 | 0 | 1000/1175 | 3 | 0 | 69.2086/69.1465 | 1.001 |
| bert-base-cased-再验证一遍 | 1 | 0 | last2h | 0 | 0 | 5 | 2e-5 | 32 | 128 | 0 | 0 | 1000/1175 | 3 | 0 | 0.7056/0.7050 | 1.80e-3 |
| bert-base-cased | 0 | 0 | last2h | 0 | 0 | 5 | 3e-5 | 64 | 128 | 0 | 0 | */440 | 4 | 0 | 69.1499/* | 0.7174 |
| bert-base-cased | 0 | 0 | last2h | 0 | 0 | 5 | 3e-5 | 64 | 70 | 0 | 0 | */440 | 4 | 0 | 69.6256/* | 0.7140 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| bert-large-uncased-wwm-squad | 0 | 0 | last2h | 0 | 0 | 2 | 7e-5 | 24 | 128 | 1/o1 | 0 | 450/470 | 4 | 0 | 69.8681/69.8477  (忘记 lowercase) | 0.7764 |
| bert-large-uncased-wwm-squad | 0 | 0 | last2h | 0 | 0 | 2 | 7e-5 | 24 | 128 | 1/o1 | 0 | */470 | 4 | 0 | 71.3203/* | 0.6745 |
| bert-large-uncased-wwm-squad | 1 | 0 | last2h | 0 | 0 | 3 | 7e-5 | 24 | 128 | 1/o1 | 0 | */700 | 4 | 0 | 71.1917/* | 1.26e-3 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

* 2nd Models
1. LSTM

|   1st-models    | SWA  |      MSD       | label-smoothing | max-seq-length | char-embed-dim | n-models | lstm-hidden-size | sentiment-dim | encode-size |  bs  |  lr  | best/total-steps | n-gpus | epochs | jaccard (best/final) |         备注         |
| :-------------: | :--: | :------------: | :-------------: | :------------: | :------------: | :------: | :--------------: | :-----------: | :---------: | :--: | :--: | :--------------: | :----: | :----: | :------------------: | :------------------: |
| roberta-base-cv |  0   | 1(p=[0.1,0.5]) |        0        |      150       |       8        |    1     |        16        |      16       |     64      | 128  | 5e-3 |     550/590      |   3    |   10   |   71.4332/71.4160    |                      |
| roberta-base-cv |  0   |       0        |        0        |      150       |       8        |    1     |        16        |      16       |     64      | 128  | 5e-3 |     550/590      |   3    |   10   |   71.4332/71.4160    |                      |
| roberta-base-cv |  0   |    1(p=0.5)    |        0        |      150       |       8        |    1     |        16        |      16       |     64      | 128  | 5e-3 |     550/590      |   3    |   10   |      71.4068/*       |                      |
| roberta-base-cv |  0   | 1(p=[0.1,0.5]) |        0        |      150       |       8        |    1     |        16        |      16       |     64      | 128  | 5e-3 |    1500/1760     |   1    |   10   |   71.8963/71.4708    |                      |
| roberta-base-cv |  0   | 1(p=[0.1,0.5]) |        0        |      150       |       8        |    1     |        16        |      16       |     64      | 128  | 5e-3 |    1500/1760     |   1    |   10   |   71.9194/71.7695    | fc 多加了一层 linear |

  

2. CNN 

   |   1st-models    | SWA  |      MSD       | label-smoothing | max-seq-length | char-embed-dim | n-models | cnn-dim | kernel-size | sentiment-dim | encode-size |  bs  | n-gpus |  lr  | best/total-steps | epochs | jaccard (best/final) |                  备注                  |
   | :-------------: | :--: | :------------: | :-------------: | :------------: | :------------: | :------: | :-----: | :---------: | :-----------: | :---------: | :--: | :----: | :--: | :--------------: | :----: | :------------------: | :------------------------------------: |
   | roberta-base-cv |  0   |    1(p=0.5)    |        0        |      150       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   3    | 4e-3 |      */295       |   5    |      71.8772/*       |                                        |
   | roberta-base-cv |  0   |       0        |        0        |      150       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   3    | 4e-3 |      */295       |   5    |      71.7584/*       |                                        |
   | roberta-base-cv |  0   |       0        |        1        |      150       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   3    | 4e-3 |      */295       |   5    |      71.7865/*       |                                        |
   | roberta-base-cv |  0   |       1        |        1        |      150       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   3    | 4e-3 |      */295       |   5    |      71.7737/*       |                                        |
   | roberta-base-cv |  0   |    1(p=0.5)    |        0        |      150       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   3    | 4e-3 |      */295       |   5    |      71.9353/*       | 当预测start 大于 end的时候返回整个原句 |
   | roberta-base-cv |  0   | 1(p=[0.1,0.5]) |        0        |      150       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   3    | 4e-3 |      */295       |   5    |      72.0055/*       |                                        |
   | roberta-base-cv |  0   | 1(p=[0.1,0.5]) |        0        |      160       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   1    | 4e-3 |      */880       |   5    |    **72.2156/***     |                                        |
   | roberta-base-cv |  0   | 1(p=[0.1,0.5]) |        0        |      150       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   1    | 4e-3 |     750/880      |   5    |   72.0859/72.0637    |                                        |
   | roberta-base-cv |  0   | 1(p=[0.1,0.5]) |        0        |      180       |       16       |    1     |   16    |      3      |      16       |     32      | 128  |   1    | 4e-3 |     750/880      |   5    | **72.2520**/72.0474  |                                        |

   