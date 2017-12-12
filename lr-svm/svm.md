# SVM 

For details about SVM, please read [Andew Ng's notes](http://cs229.stanford.edu/notes/cs229-notes3.pdf) or Hang Li's statistical learning book in Chinese.
 
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

[1] [Andrew Ng's page 14-15](http://cs229.stanford.edu/notes/cs229-notes3.pdf)

## Advantage and disadvantage of SVM

### Advantage
* better performance for small size of data.
* good generalization 
* no local minimal(it's a convex optimization, local optimum  means global optimum??)
* handle high-demenisonal data
* have a solid theory to support. 

### Disadvantage: 
* Kernel selection and parameter tuning.
* Originally for binary classification, not for multi-classification 
* High algorithmic complexity and extensive memory requirement required by quadratic programming in large-scale tasks. 


## What's SMO(sequential minimal optimization)?

SMO is an iterative algorithm for solving  SVM dual problem. SMO breaks this problem into a series of smallest possible sub-problems, which are then solved analytically. Because of the linear equality constraint involving the Lagrange multipliers $$\alpha_i$$, the smallest possible problem involves two such multipliers. Then, for any two multipliers   $$\alpha_{1}$$ and $$\alpha_{2}$$ , the constraints are reduced to:
e reduced to:


$$
0 \leq \alpha_1, \alpha_2 \leq C
$$

$$
y_1\alpha_1+y2\alpha_2 = k
$$
and this reduced problem can be solved analytically: one needs to find a minimum of a one-dimensional quadratic function. $$k$$ is the negative of the sum over the rest of terms in the equality constraint in the dual problem, which is a constant.

Steps:
1. Find a Lagrange multiplier $$\alpha_1$$ that violates the KKT conditions for the optimization problem.
2. Pick a second multiplier $$\alpha_2$$ and optimize the pair $$(\alpha_1, \alpha_2)$$ (pick two that allow us to make the biggest progress towards the global maximum)
3. optimize $$W(a)$$ with respect to $$\alpha_1$$ and $$\alpha_2$$, while holding all the other $$\alpha_k$$'s$$(k \neq 1,2)$$ fixed.
4. Repeats steps 1 and 2 until converge.

[1][http://cs229.stanford.edu/notes/cs229-notes3.pdf](http://cs229.stanford.edu/notes/cs229-notes3.pdf)

[2][https://en.wikipedia.org/wiki/Support_vector_machine](https://en.wikipedia.org/wiki/Support_vector_machine)