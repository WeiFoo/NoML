# Train/dev/test/distribution

Choose a dev set and test set to reflect data you expect to get in fth future and cosindier important to do well on.

# Size of the dev and test sets

Traditionally, train/test: 70%, 30%; or train/dev/test:60%, 20%, 20%

Currently, train/dev/test: 99%, 1%, 1% for deep learning algorithm.

Size of test set: set your test set to be going enough to give high confidence in the overall performance of your system.

# When to change dev/test sets and metrics?

If doing well on your metric + dev/test set don't correspond to doing well on your application, change your metric and/or dev/test set.

# Summary of bias/variance with human-level perforamnce

* Huan-level error: as a proxy for Bayes error

* Training error:

  * training error - human -level error = "Avoidable bias"
  * Training bigger modeel
  * Train longer/better optimization algorithms 
  * NN architecture/hyperparameters search

* Dev error :

  * Dev error - Training error = "variance "
  * More data
  * Regularization: L2, dropout, data augmentation, 
  * NN architecture/hyperparameters search.

# Two fundamental assumptions of supervised learning

* You can fit the training set pretty well.

* The training set performance generalizes pretty well to the dev/test set.

# Error analysis

# Build your system quickly

* Set up dev/test set and metric

* Build inital system quicily 

* Use Bias/Variance analytiss & Error analytiss ot prioritize next steps. 

# Bias/Variance on mismatched training and dev/test sets
![](/assets/Screen Shot 2017-11-03 at 00.20.49.png)




