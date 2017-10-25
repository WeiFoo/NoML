# Performance Metrics

## How to draw ROC?

```
In signal detection theory, a receiver operating characteristic (ROC), or simply ROC curve, is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied.

作者：zhwhong
链接：http://www.jianshu.com/p/c61ae11cc5f6
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出```



Most binary classifiers give a prediction probability for positive and negative classes. If you set a threshold say, 0.6, you will get a Recall(TPR) and False alarm(FPR). then you vary this threshold value, you will get a group of points. threshold value = 0 corresponds to the point (1,0) while threshold value = 1 corresponds to point(0,0)

![](http://upload-images.jianshu.io/upload_images/145616-2063bb79c3684a8a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
[1] [机器学习之分类性能度量指标 : ROC曲线、AUC值、正确率、召回率](http://www.jianshu.com/p/c61ae11cc5f6)


## Why use AUC?
```
he AUC value is equivalent to the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

作者：zhwhong
链接：http://www.jianshu.com/p/c61ae11cc5f6
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。 
```