# Why Padding?
* If we don't use padding, we end up shrinking the size of output. With padding, we can keep input and output as the same size.
* Without padding, the pixels near the edge will have fewer contribution than pixels in the middle of the picture. That said, without padding, we might throw information from edge.

# Input size(n), conv filter size(f), and output size(o) 
* without padding, $$o = n-f+1$$
* with padding, $$o = n+2p-f+1$$, where $$p$$ is the size of padding.
note: when $$p=\frac{f-1}{2}$$, then the output size = input size

# Strided convolution
input: $$n * n$$
filter: $$f * f$$
padding: $$p * p$$
stride: $$s * s$$
output: $$\lfloor\frac{n+2p-f}{s} +1\rfloor * \lfloor\frac{n+2p-f}{s} +1\rfloor$$