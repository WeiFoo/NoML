# Performance Metrics

## How to draw ROC?


In signal detection theory, a receiver operating characteristic (ROC), or simply ROC curve, is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied.


Most binary classifiers give a prediction probability for positive and negative classes. If you set a threshold say, 0.6, you will get a Recall(TPR) and False alarm(FPR). then you vary this threshold value, you will get a group of points. threshold value = 0 corresponds to the point (1,0) while threshold value = 1 corresponds to point(0,0)

![](http://upload-images.jianshu.io/upload_images/145616-2063bb79c3684a8a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
[1] [机器学习之分类性能度量指标 : ROC曲线、AUC值、正确率、召回率](http://www.jianshu.com/p/c61ae11cc5f6)


## Why use ROC/AUC?

The AUC value is equivalent to the probability that a randomly chosen positive example 
i s ranked higher than a randomly chosen negative example.
When data sets are imbalanced, ROC/AUC is more stable than Recall, F1, precision..


## Confusion matrix


$$ Recall = TP/(TP+FN)$$
$$Precision = TP/(TP+FP)$$
$$F1 = 2 Recall* Precision /(Recall + Precision)$$
$$PF = FP/(FP+TN )$$
$$Acc = (TP+TN)/(TP+TN+FP+FN)$$