# Meta algorithms

Bagging, boosting, stacking all are so-called ''meta-algorithms'': Approach to combine serval machine learning techniques into one predictive model in order to decrease the __variance__ (bagging), __bias__ (boosting) or improve the __predictive force__(stacking alias ensemble).


* __Bagging__ (stands for **Bootstrap Aggregation**, parallel ensemble) is the way decrease the variance of your prediction by generating additional data for training from your original dataset using combinations with repetitions to produce multisets of the same cardinality/size as your original data(__samples are drawn with replacement__). By increasing the size of your training set you can't improve the model predictive force, but just decrease the variance and helps to avoid overfitting, narrowly tuning the prediction to expected outcome.
    * example: __random forest__ 
* __Boosting__(sequential ensemble) is a two-step approach, where one first uses __subsets of the original data__ to produce a series of averagely performing models and then "boosts" their performance by combining them together using a particular cost function (=majority vote). Unlike bagging, in the classical boosting the subset creation is not random and depends upon the performance of the previous models: every new subsets contains the elements that were (likely to be) misclassified by previous models.
    * example: __Aaboost__
* __Stacking__ is a similar to boosting: you also apply several models to your original data. The difference here is, however, that you don't have just an empirical formula for your weight function, rather you introduce a meta-level and use another model/approach to estimate the input together with outputs of every model to estimate the weights or, in other words, to determine what models perform well and what badly given these input data.

[1] [https://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning](https://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning)
### Bagging

Given a standard training set $$D$$ of size n, bagging generates m new training sets $$D_{i}$$, each of size n′, by sampling from D uniformly and with replacement. By sampling with replacement, some observations may be repeated in each $$D_{i}$$. If $$n′=n$$, then for large n the set $$D_{i}$$ is expected to have the fraction $$ (1 - 1/e)(≈63.2\% )$$ of the unique examples of D, the rest being duplicates This kind of sample is known as a __bootstrap sample__. The $$m$$ models are fitted using the above $$m$$ bootstrap samples and combined by __averaging the output__ (for regression) or __voting__ (for classification)

[1] [https://en.wikipedia.org/wiki/Bootstrap_aggregating](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
### Boosting

__Incrementally building an ensemble by training each new model instance to emphasize the training instances that previously mis-classified.  In some cases, boosting has been shown to yield better accuracy than bagging, but it also tends to be more likely to over-fit the training data__

Most boosting algorithms consist of iteratively learning weak classifiers with respect to a distribution and adding them to a final strong classifier. When they are added, they are typically weighted in some way that is usually related to the weak learners' accuracy. **After a weak learner is added, the data are reweighted**: examples that are misclassified gain weight and examples that are classified correctly lose weight (some boosting algorithms actually decrease the weight of repeatedly misclassified examples, e.g., boost by majority and BrownBoost). Thus, future weak learners focus more on the examples that previous weak learners misclassified.
  
  
  

