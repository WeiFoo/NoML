# SVM 

For details about SVM, please read [Andew Ng's notes](http://cs229.stanford.edu/notes/cs229-notes3.pdf) or Hang Li's statistical learning in Chinese.
 
## Why do we solve dual problem not prime problem in SVM?
* The complexity of the optimization algorithm is related the dimensionality $$D$$ in the prime problem; in the dual problem, such complexity is related to the number of samples in training data set, $$N$$. If is's a linear classification, and $$D \leq N$$ , we could solve the prime problem directly. or SGD.
* Dual problem is easier to optimize when the data is nonlieaner inseparable, We could apply kennel trick.
* When predicting, if optimizing dual problem we get $$a_i$$, then the prediction can be written as:
  $$
   w^Tx+b = (\sum_i a_i y_i x_i)^Tx_j+b = \sum_i a_i y_i < x_i, x_j> + b \:\:\:\:(1)
  $$
  That means we only need to calculate the inner product between test data and training data. Moreover, some $$a_i$$'s will be zero if they are not support vectors.
  
  
 [1] [Andew Ng's note ](http://cs229.stanford.edu/notes/cs229-notes3.pdf)
 
## Why use kernel function in SVM ?
The kernel trick avoids the explicit mapping that is needed to get linear learning algorithms to learn a nonlinear function or decision boundar
In SVM, if we want to map our features into higher dimension, then we need to find a mapping function$$\phi$$ to do that. Then in that higher dimension feature space, we find a linear boundary to classify the data. For our dual problem, 
we have inner product format in function (1), then we define the kernel as $$K(x,z)= \phi(x)^T\phi(z)$$. The interesting is that, often K(x,z) may be very __inexpensive__ to calculate, even though $$\phi(x)$$ itself may be very __expensive__ to calculate(Perhaps because it's an extremely high dimensional vector). If we get an kernel functionK(x,z), then we can get SVM to learn in ghih dimensional feature space given by $$\phi$$, but whitough even having to explicitly find or represent vectors $$\phi(x)$$ 

[1][Andrew Ng's page 14-15](http://cs229.stanford.edu/notes/cs229-notes3.pdf)