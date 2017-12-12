# Randome Forest

## Ensemble Learners
Random Forest is an Ensemble Learner(Bagging, not Boasting). 
Basically, there are 3 types of ensemble learners, i.e. Bagging, Boosting and Stacking. 
A short video explanation can be found here:
- [Ensemble Learners](https://www.youtube.com/watch?v=Un9zObFjBH0)  [2](https://www.youtube.com/watch?v=062w-dGDRr0)
- [Bagging](https://www.youtube.com/watch?v=2Mg8QD0F1dQ), reduce variance.
- [Boosting](https://www.youtube.com/watch?v=GM3CDQfQ4sw), reduce bias.
- [Stacking], combine several weak learners(often diffrent in nature) and make it a stronger learner 
by finding the best way to weight each weak learner.


## Pros
- The bias remains same as that of a single decision tree. 
However, the variance decreases and thus we decrease the chances of overfitting.
- A quick and dirty way out, random forest comes to the rescue. 
Don't have to worry much about the assumptions of the model or linearity in the dataset. 

## Cons
- Random forests don't train well on smaller datasets.
- There is a problem of interpretability with random forest.
- The time taken to train random forests may sometimes be too huge.
- In the case of a regression problem, the range of values response variable can 
take is determined by the values already available in the training dataset. 

## Refs
Pros and Cons are based on [this post](https://www.datasciencecentral.com/profiles/blogs/random-forests-explained-intuitively).

