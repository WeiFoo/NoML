 # Gradient Boosting Decision Tree
 ![Example](http://arogozhnikov.github.io/images/gbdt_attractive_picture.png "Example")
 
 ### Visualization
 A great visualization and playground of Decision Tree and Gradient Boosting could be found [here](http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)
 
 
 ### Regression Decision Tree
 >Waiting to be written.
 
 
 ### Boosting  Decision Tree
 >Waiting to be written.
  
 ### Gradient Boosting Decision Tree
 Gradient boosting builds an ensemble of trees one-by-one, 
 then the predictions of the individual trees are summed:
 ```    
     D(x) = d_tree1(x) + d_tree2(x) + ...
 ```
 The next decision tree tries to cover the discrepancy between the target function `f(x)` and 
 the current ensemble prediction by reconstructing the residual.
 For example, if an ensemble has 3 trees the prediction of that ensemble is:
 ```   
     D(x) = d_tree1(x) + d_tree2(x) + d_tree3(x)
 ```
 The next tree `tree_4` in the ensemble should complement well the existing trees and 
 minimize the training error of the ensemble. In the ideal case we'd be happy to have:
 ```
     D(x) + d_tree4(x) = f(x)
 ``` 
 To get a bit closer to the destination, we train a tree to reconstruct the difference between 
 the target function and the current predictions of an ensemble, which is called the residual:
 ```
     R(x) = f(x) - D(x)
 ```
 Did you notice? If decision tree completely reconstructs `R(x)`, the whole ensemble gives predictions 
 without errors (after adding the newly-trained tree to the ensemble)! That said, in practice 
 this never happens, so we instead continue the iterative process of ensemble building.
  
 **Source**: [http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html](http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)
