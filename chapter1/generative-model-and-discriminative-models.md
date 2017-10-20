Generative v.s. Discriminative Models

## What's Generative or Discriminative model?

Let's say you have input data x and you want to classify the data into labels y. A generative model learns the joint probability distribution $$p(x,y)$$ and a discriminative model learns the conditional probability distribution $$p(y|x)$$ - which you should read as "the probability of $$y$$ given $$x$$".

Generative algorithms model p(x,y), which can be tranformed into p(y|x) by applying Bayes rule and then used for classification.

The overall gist is that discriminative models generally outperform generative models in classification tasks.

A generative algorithm models how the data was generated in order to categorize a signal. It asks the question: based on my generation assumptions, which category is most likely to generate this signal? A discriminative algorithm does not care about how the data was generated, it simply categorizes a given signal.

## Give examples on Generative and Discriminative model

Generative:  Gaussian Naive Bayes

Discriminative: Logistic regression 

[1] [https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm](https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm)