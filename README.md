# CSE-40535-Vision

Crystal Colon: **ccolon2**  
Chase Brown: **cbrown54**

https://www.kaggle.com/grassknoted/asl-alphabet

https://colab.research.google.com/drive/14cnjsezX2TxjJ1ZFu8WJ5hyMswmmxB_H?usp=sharing

Description:
A.)The dataset contains 87,000 200x200 images of the ASL alphabet. There are 29 classes, of which 26 are for the letters A-Z and 3 classes are for SPACE, DELETE, and NOTHING.

B)  
   The data is already nicely split into testing and training data, however, there is only one photo for each of the 29 classes in the test folder.  Meanwhile there are 3000 photos for each class in the training folder.  We will have to move a couple hundred photos from the training data into the testing data for each class and furthermore split the data testing data into two groups, validation and testing.
  
After looking at several images we believe key features we should focus on identifying are knuckles and finger nails.  depending on if you extend your fingers or scrunch your hand in a fist, your knuckles can be the darkest or the lightest area on your hand.  A large portion of the photos are taken with no foreground light and you can only see the silhouette of the hand.  As far as we are aware, the only option in this case would be to indentify shapes by the outline of the hand.  

Another idea is that your fingers will form straight lines when two or more fingers are aligned next to eachother.  We may be able to create a filter which identifies straight lines well.  

Our last piece of information we noticed is the importance of the thumb in ASL.  We believe it will help greatly if we can identify the position of the thumb.  We have noticed the finger nail of the thumb is generally perpendicular to the rest of your finger nails.  We are not sure of any strategies to help identify this feature, but if there are any, we believe it might be important.

**Image Loading**  
We did not do any manual data collection, so much of our data was already normalized. Each image comes from our datasource as 200x200. This is already a pretty good resolution for our future work, however, we did implement in our image reader a way to resize the images again. The reason we chose to implement this functionality is incase we attempt to implement some sort of a Convoluntional Neural Network as they do not need images with such high resolution and it will likely show diminishing returns above 64x64.  The final piece of getting our images setup is splitting them into their perspective categories, training, testing, and evaluating. The dataset comes with a unique folder of data for evaluating so all we had to do was split the training data into training and testing. We use the data split function from scikit to do a 9:1 split across the training data.  Note 'J' and 'Z' have a finger trace in their letter so we simply chose to use the last frame of the video to be the symbol for those two letters.

**PreProcessing**  
Our preprocessing is simple because the dataset was in such good condition to get started. The two pieces of preprocessing we were able to was first converting the labels into one hot vectors. This was a process learned in Natural Language Processing where we are converting the string labels into an identity matrix. This speeds up training time and allows the learning of categorical data.  The second piece of preprocessing we did was to simply normalize the RGB values.  This will remove the distortion caused by lights and shadows to help us better identify edges of fingers and orientations of the hand moving forward.  

**Feature Extraction**  
One method for feature-extraction that we attempted was texture-based. We computed Haralick Texture features. The core concept behind computing Haralick features is the Gray-Level Co-occurence Matrix, which uses the adjacency concept in pictures.
The matrix looks for pairs of adjacent pixel values that occur in the picture and keeps recording it over the entire picture.  

This method was tested with two different SVM classifiers (linear and rbf). We chose these two classifiers immediately because they were fresh in our brain from the previous practical.
Linear worked pretty well. However, the first image that this classifier was evaluated on was immediately wrong, but not by a lot. For example, the “A” letter in sign language was wrongfully interpreted as an “S” (image result included below). Upon comparing the two letters, I realized that the shape of the hand was very similar, so it is reasonable that this misclassification occured between the letter A and S.  
![alt text](https://github.com/chaseBrow/CSE-40535-Vision/blob/main/pics/example_image.jpg?raw=true)  

I found similar result with rbf (one image mispredicted but not in the same way). We did not test with poly because linear and rbf both performed moderately with classification accuracies reaching 43% and 47% respectively. Both linear and rbf SVMs had similar issues where they would work extremely well for certain ASL Characters (reaching ~80% classification accuracy on a single character) and then falling short on other characters (~5% classification accuracy). We made the assumption a poly SVM would have similar outcomes and wanted to move on to another approach before committing the computing resources to testing poly. We found the SVM program takes a long time to run with the size of the dataset (15-20min/epoch although this could be because we are running it locally on our laptops whereas the CNN approach to follow we run on Google Colab using a GPU accelerator). Ultimately, we did not decide to go with the texture-based feature extraction method because it was not as accurate as we saw many papers reaching online.   
  
**The CNN Approach**  
With a CNN, we don’t need to manually extract features from our images. The network automatically extracts features and learns their importance on the output by applying weights to its connections. By our understanding CNNs do the work of both feature extractors and classifiers. 
![alt text](https://github.com/chaseBrow/CSE-40535-Vision/blob/main/pics/ftExt.png?raw=true)
![alt text](https://github.com/chaseBrow/CSE-40535-Vision/blob/main/pics/cnn.png?raw=true)
In all honesty, it feels a little like cheating. A CNN is simple to define with Keras, has many predefined training and fitting functions, and provides much more accurate results (>97% in our case). After seeing the results of the first simple CNN (2 Conv2d layers, 1 MaxPool, and 1 Dense layer) we knew a CNN was going to be much more efficient and we were able to work with it easily in Google Colab giving us access to GPUs.  
Seemingly the best part of being part of using a CNN was how quickly we were able to make modifications to our model and make small adjustments to the kernel size and other parameters to see the impacts they had on the output.  After reading a few papers and testing out some of their proposed layer configurations we settled on the following.  
(NOTE: We also discovered this really cool application called Netron https://netron.app/ where you are able to upload your saved models and visualize them, we included a photos in the pics folder)  
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================  
conv2d_1 (Conv2D)            (None, 64, 64, 16)        448       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 21, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 21, 21, 32)        9248      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 21, 21, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 7, 7, 128)         73856     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 7, 7, 256)         295168    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 256)         0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 2, 2, 256)         1024      
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_2 (Dense)              (None, 29)                14877     
=================================================================  
Total params: 942,557  
Trainable params: 942,045  
Non-trainable params: 512  

**CNN Explained**  
The sequence of CONV2D and MaxPooling layers is simply the creation and connection of convolutional kernels. We found the final couples of layers to be more impactful in the accuracy of the model. First the batch_normalization layer, this layer is interesting as it behaves differently during training VS inference. The key idea is that it maintains the mean output close to 0 while standard deviation close to 1. During training it normalizes using the mean and stdev of the current inputs while in inference it uses a moving average of the mean and stdev of the training batch. This normalization stabalized our learning significantly and reduced the number of training epochs required from 15 down to 5.  
We used a dropout layer to randomly assign 0 values to certain inputs in order to prevent overfitting.  (NOTE: no data will be dropped during inference)  
Finally in the final 2 dense layers where all of the input data is fed into each of the hidden neurons. We found these layers to be the two most common layers in all of the examples and papers we read. We say fluctuation in the amount of CONV2D and MaxPooling layers, but, this combination of dense layers was extremely prevelant.  

**TRAINING**  
Epoch 10/10  
74385/74385 [==============================] - 15s 197us/step - loss: 0.1287 - acc: 0.9889 - val_loss: 0.0819 - val_acc: 0.9948  

**EVALUATING**  
4350/4350 [==============================] - 0s 107us/step  
Evaluation Accuracy =  99.52%   
Evaluation loss =  0.079331  

**Next Steps**
After speaking with several of our peers about their implementation of CNNs for their projects we were shocked with how accurate ours was. From our project we had the impression CNNs were extremely capable and unbeatable. With achieving classification accuracies of over 99% we did not think there was room for improvement in our model. We have come to the understanding it is not completely due to the power of CNNs though, it is partly to thank for our dataset. Our dataset is quite large and contains plenty of data for training and testing. However, ASL Characters are much more precise than a written character, because of this there is little variation in the images apart from the lighting of the images. People learn to trace certain characters in unique ways which makes written character classification a slightly harder challenge. However, ASL is standardized and confined to the movements of a hand which are limited compared to that of a pen on paper. We would propose to better validate our model we should attempt to evaluate it on a larger dataset of our hands (because neither of us know how to sign ASL and we will likely have small errors in our character formation) to provide it with a more varried evaluation set. We would also be interested to see if it can detect 3-4 character words from video or if this will simply identify hundreds of different characters during the transition from word to another.

Citations:  
https://www.irjet.net/archives/V7/i11/IRJET-V7I1155.pdf  
http://ijcsit.com/docs/Volume%205/vol5issue01/ijcsit2014050166.pdf  
https://www.kaggle.com/paultimothymooney/interpret-sign-language-with-deep-learning  
https://freecontent.manning.com/the-computer-vision-pipeline-part-4-feature-extraction/  
