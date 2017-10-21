# Adaboost

![http://www.csuldw.com/assets/articleImg/adaboost-algorithm.png](http://www.csuldw.com/assets/articleImg/adaboost-algorithm.png)

### Steps

1. Initialize very sample weight to $$\frac{1}{N}$$
2. Iterate M times. Each time, change the  the training data weights according to training error rates $$e_m$$. The basic rules are: increases the weights for mis-classified data and reduce the weights of correctly classified data.
3. According to the weights of all weak learners $$a_m$$, combine all $$M$$ learners together, then finally give a output as $$G(X) = \sum_{m=1}^{M}a_mG_m(x)$$ 

[1] [Adaboost - 新的角度理解权值更新策略 in Chinese](Adaboost - 新的角度理解权值更新策略) 
[2] [Adaboost Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)

