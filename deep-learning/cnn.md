# Why Padding?
* If we don't use padding, we end up shrinking the size of output. With padding, we can keep input and output as the same size.
* Without padding, the pixels near the edge will have fewer contribution than pixels in the middle of the picture. That said, without padding, we might throw information from edge.

# Input size(n), conv filter size(f), and output size(o) 
* without padding, $$o = n-f+1$$
* with padding, $$o = n+2p-f+1$$, where $$p$$ is the size of padding.
note: when $$p=\frac{f-1}{2}$$, then the output size = input size

# Strided convolution
input size: $$n * n$$
filter size: $$f * f$$
padding size: $$p * p$$
stride size: $$s * s$$
output size: $$\lfloor\frac{n+2p-f}{s} +1\rfloor * \lfloor\frac{n+2p-f}{s} +1\rfloor$$

# What's MaxPooling?

Divide the input into regions as the same size as maxpooling filter.
So what the max operation does is so long as the feature is detected anywhere in one of these quadrants, it then remains preserved in the output of Max pooling. So what the max operator does is really it says, if the feature is detected anywhere in this filter, then keep a high number. But if this feature is not detected, so maybe the feature doesn't exist in the upper right hand quadrant, then the max of all those numbers is still itself quite small.

Parameters: filter size **f** and stride **s**. The output size of pooling will be the same as strided convolution output. Typically, f = 2, s = 2; f = 3, s = 2; Mostly, maxpooling doesn't use padding.

No parameters to learn in backprob.

