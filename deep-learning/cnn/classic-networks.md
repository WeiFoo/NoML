# LeNet - 5
#### Structure:

![](/assets/Screen Shot 2017-12-02 at 15.13.22.png)

#### Dimensions in each Layer 
* conv(f=5x5,s=1)
* pooling(f=2,s=2)
* conv(f=5x5,s=1)
* pooling(f=2,s=2)
* fully connected layer(120)
* fully connected layer(84).

5 layers: 2 convolutional layers, 2 fully connected layers, 1 output layer
Totally, 60K parameters need to learn.


# Alex - Net

![](/assets/Screen Shot 2017-12-02 at 15.12.59.png)
7 layers: 4 convs, 2 fc, 1 output
Totally, 60 M parameters.

# VGG - 16

![](/assets/Screen Shot 2017-12-02 at 15.24.38.png)
convs: 3x3, stride = 1,
max-pooling: 2x2, stride = 2,
16 layers: 13 convs, 2 fc, 1 output
Totally, 138M parameters.

# ResNets
Skip connections, can help much deeper networks.
![](/assets/Screen Shot 2017-12-02 at 15.38.15.png)

![](/assets/Screen Shot 2017-12-02 at 15.39.05.png)

# Inception Neworks(GoogleNet)

![](/assets/Screen Shot 2017-12-02 at 16.13.04.png)

Using 1x1 convolution to reduce computational cost
  
![](/assets/Screen Shot 2017-12-02 at 16.03.59.png)

![](/assets/Screen Shot 2017-12-02 at 16.15.11.png)

![](/assets/Screen Shot 2017-12-02 at 16.15.52.png)
