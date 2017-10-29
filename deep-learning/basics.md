# Normalizing inputs

* Zero out or subtract out the mean of your every data: $$X-=\mu$$
* Each data divided by sample variance: $$X/=\sigma^2$$
* Use the same $$\mu$$ and $$\sigma^2$$ values to normalize your testing data as well. you don't need to estimate the new values for both of them. Just make sure your training and testing data go through the same transformation.


[1] [Andrew Ng's Coursera video ](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs)

# Why normalizing inputs?

Make all your features on the same scale, hopefully will make your loss function or cost function easier to optimize, especially when using SGD to optimize. See the picture.
![](/assets/Screen Shot 2017-10-28 at 10.14.07 PM.png)

[1] [Andrew Ng's Coursera video ](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs)

