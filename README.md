# Cracks-Image-Classification

## Abstract

<p align="justify"> This work aims to represent a classification of craks or nocraks using deep learning
techniques. Basically, train a model that classifies a set of images separating at 1 for nocracks
and 0 for cracks. Using convolutional networks, also of multilayer perceptron, to avoid
overfitting. And test with different parameters to find a better solution for our model.</p>

## Introduction

<p align="justify"> The image classification presented in this research is intended to classify an image
with cracks or without cracks. Image classification is a concept of computer vision that
can classify images through human standards. It is a very easy task for a person to
distinguish but for a machine it is a task with a certain degree of complexity.
Image classification is a supervised learning program where you give the inputs and
the desired target to your train. Among the different types of artificial neural networks,
one will be used specifically convolutional neural networks (CNNs). CNNs [4] basically
processes inputs generating an intermediate representation, then abstracts layers by layers
until it reaches a fully connected layer at the end of the process, which in this case is the
outputs. </p>
<p align="justify"> For coding it will be done in the Pytorch [3] framework. Pythorch launched in
October 2016, is a low-level API focused on working directly with array expressions. Its
programming language is already perceived by the name that is Python, a high level
language. Pytorch has a wide variety of libraries, functions, it is simple to learn and very
efficient. It is a powerful framework however it is becoming a preferred solution for
academic research and deep learning applications that require personalized expressions. </p>

## Dataset

<p align="justify"> At the moment, the databases have a total of 310 images that are being prepared for
training and testing. The distribution of the data set is as follows: 95% to train the
algorithm and then create the predictive model and 5% for tests. Considering the size of
the 480 x 320 images. </p>


![dataset nocracks](https://user-images.githubusercontent.com/38785749/92022603-6b826e80-ed53-11ea-8a93-c383591fed1c.png)

## Implementation 

<p align="justify"> Pytorch came as a solution for people frustrated by the other structures that have evolved without giving greater flexibility in their projects. The Pytorch with its dynamic form has benefited, enhanced with speed and efficiency, providing a better result for these people.
Now, let's start coding by importing the torch and the other necessary libraries: </p>


    import torch

    import torchvision

    import numpy as np

    from torchvision import transforms

    from PIL import Image

    from os import listdir

    import random

    import torch.optim as optim

    from torch.autograd import Variable

    import torch.nn.functional as F

    import torch.nn as nn

    from torch.nn.utils.rnn import pad_sequence

    import time


<p align="justify"> Creating the standardization of data with transforms.Normalize that have three parameters mean, std and inplace. The mean is the sequence of averages for each channel. Std is the sequence of standard deviations for each channel and inplace is a Boolean variable and also an optional default is false, in order to perform operation in place (true). Another common image transformation method is transforms.Compose which has a list as a parameter. </p>

    normalize = transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

    transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(256),

        transforms.ToTensor(),

        normalize])
      
    
<p align="justify"> At this stage, the data directory folder is being loaded. A folder with the data of cracks and nocracks that are separated by labeling. In this case, 1 corresponds to the data of nocracks and 0 the data of cracks. </p>


    files = listdir('images/train')

    for i in range(len(listdir('images/train'))):

        f = random.choice(files)

        files.remove(f)

        img = Image.open("images/train/" + f)

        img_tensor = transform(img)


        train_data_batch.append(img_tensor)
        nocracks = 1 if 'nocracks' in f else 0
        cracks = 1 - nocracks

        target = [cracks, nocracks]
        target_list.append(target)


        if len(train_data_batch) >= batchSize: 
            train_data.append((train_data_batch, target_list))

            train_data_batch = []
            target_list = []

    
        
    
<p align="justify"> In this feedforward model, sigmoid neurons are being used, their output is a probability of a real value between 0 and 1. And also, with Conv2d convolutional networks being applied with some parameters, for example, the input channel receives from 3.6, 12,18 and output 6,12,18,24 for each pair (an input and output channel) for different neurons. The kernel size 3 and padding equal to 1. The activation function is Relu, returning x for all values of x > 0 and returning 0 for all values where x < 0 and max_pool2d to reduce the number of dimensions of the resource map. </p>


    class Net(nn.Module):

        def __init__(self):

            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(3,6,kernel_size=3,stride=1,     padding=1)

            self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride = 1, padding=1)

            self.conv3 = nn.Conv2d(12, 18, kernel_size=3, stride = 1, padding=1)

            self.conv4 = nn.Conv2d(18, 24, kernel_size=3, stride = 1, padding=1)

            self.fc1 = nn.Linear(6144, 60) 

            self.fc2 = nn.Linear(60, 2) 


        def forward(self, x):

            x = self.conv1(x)

            x = F.max_pool2d(x, 2)

            x = F.relu(x)

            x = self.conv2(x)

            x = F.max_pool2d(x, 2)

            x = F.relu(x)

            x = self.conv3(x)

            x =F.max_pool2d(x, 2)

            x = F.relu(x)

            x = self.conv4(x)

            x =F.max_pool2d(x, 2)

            x = F.relu(x)        

            x = x.view(x.size(0), -1)

            x = F.relu(self.fc1(x))

            x = self.fc2(x)

            return torch.sigmoid(x)


<p align="justify"> Since the algorithm was executed on a GPU, it is extremely important to call the CUDA method on the model. The following code snippet tells Pytorch to run the algorithm on the GPU: </p>


    if torch.cuda.is_available():
        model.cuda()

<p align="justify">Using the Adam model for optimization. Adam is basically an equation for updating a model's parameters. </p>

    optimizer = optim.Adam(model.parameters(), lr=0.01)

<p align="justify"> We set the gradients to zero before we do the backpropagation because in pytorch it accumulates the gradients in each pass of a sequence. </p>

    optimizer.zero_grad()
 
<p align="justify"> Then, the Binary-Cross-Entropy Loss was coded, which simply calculates the loss of the classification network by predicting the possibilities, in which it should be summarized as 1 for nocracks and 0 for the presence of cracks. </p>

    criterion = F.binary_cross_entropy


<p align="justify"> The command to train the network: </p>

    net.train()

<p align="justify"> The loss.backward () call accumulates the gradients for each parameter and, for this reason, it uses optimizer.zero_grad (), that is, it avoids accumulating the gradients of several passages. </p>

    loss.backward()

<p align="justify"> Then, using .step, we update the parameters based on the current gradient. </p>

    optimizer.step()

<p align="justify"> Finally, we implemented the test method, in which we take new images from the test folder. </P>


  
    def test(): 

        model.eval()

        files = listdir('images/test/')

        f = random.choice(files)

        img = Image.open("images/test/" + f)

        img_eval_tensor = transform(img)

        img_eval_tensor = img_eval_tensor.unsqueeze(0)

        data = Variable(img_eval_tensor) 

        out = model(data)

        print(out.data.max(1, keepdim=True)[1])

        img.show()

        time.sleep(5)

    for epoch in range(1, 30):

        train(epoch)

        test()

## Tests 

<p align="justify"> With training the model reached an accuracy of 0.70 with loss 0.688693 in 14 epochs. Testing our model with different images we had an accuracy of 71.43%, hitting 10 out of 14 images. </p>
