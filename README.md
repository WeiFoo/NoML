# NoML--No Machine Learning

No matter how you end up reading this post, if you will interview machine learning engineer or data scientist job,
you must be well prepared for that. Do you know machine learning? How well?  Consider the following questions:

```
*  What's bias and variance?
*  What's L1 and L2 regularization? which is more stable?
*  What's relationship between logistic regression and SVM?
*  What's SVM? please derive it mathematically.
*  How to update logistic regression parameters? which algorithms? 
*  Do you know deep learning? please write a basic fully-connected neural 
   network with numpy using any activation function?

```
If you could answer those questions pretty well(Chinese "pretty well", not....), I think you're ready for the interview. 
Machine learning interview questions can be asked as simple as explaining confusion matrix, drawing ROC, or as hard as implementing K-means, basic neural networks or even some mathematical explanation of SVM. These are really basic, not even considering practical issue. So, how do you prepare?

You might think that we can read a ML book, why do we need to create this notebook? Because sometimes, when we read a book, we get lost and  don't know what kind of question will be asked.

So the beauty of this book is to **collect questions or asked questions!**

## How to Contribute

Please fork this repo and submit pull requests. In this repo, we use Markdown and [Katex](https://khan.github.io/KaTeX/) to edit the notes.



Basically, there're two formats: question-answer and note. You can either ask a question and answer it, like the following example

```markdown
Q: What's overfitting?

A: Overfitting refers to a model that models the training data too well. Overfitting happens
   when a model learns the detail and noise in the training data to the extent that it negatively
   impacts the performance of the model on new data.

R: https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/
```
OR write a note about specific algorithm, like:

```
# Meta algorithms

Bagging, boosting, stacking all are so-called ''meta-algorithms'': Approach to combine
serval machine learning techniques into one predictive model in order to decrease the 
variance(bagging),bias(boosting) or improve the predictive force(stacking alias ensemble).

```


Then if necessary, you can give an explanation. Please note, **you can asked any questions(write any notes) you want.  But the answer you provided should always come with a link\(give credits to the original author\).  **At least, let the readers judge how good the answer is and whether it makes sense!


## Read at GitBook

This project is hosted at gitbook, you can read it here!==&gt;[https://weifoo.gitbooks.io/noml/content](https://weifoo.gitbooks.io/noml/content/)



