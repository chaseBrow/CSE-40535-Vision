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
One method for feature-extraction that we attempted was texture-based.We computed Haralick Texture features. The core concept behind computing Haralick features is the Gray-Level Co-occurence Matrix, which uses the adjacency concept in pictures.
The matrix looks for pairs of adjacent pixel values that occur in the picture and keeps recording it over the entire picture.  

This method was tested with two different types of kernels: linear and rbf.
Linear worked pretty well. However, the first image that this algorithm was tested on was immediately wrong, but not by a lot. For example, the “A” letter in sign language was wrongfully interpreted as an “S” (image result included below). Upon comparing the two letters, I realized that the shape of the hand was very similar, so it is reasonable that this misclassification occured.  

I also got a similar result with rbf (one image mispredicted but not in the same way).We did not test with poly because linear (the most simple kernel) already gave decent results. Also, this program takes a long time to run with the size of the dataset. Ultimately, we did not decide to go with the texture-based feature extraction method because it was less accurate and did not render as fast results. With a CNN, we don’t need to manually extract features from an image. The network automatically extracts features and learns their importance on the output by applying weights to its connections. CNNs are both feature extractors and classifiers. Because using a CNN allows you to accomplish more with less and yields a more accurate result (>97%), we used this method. 
![alt text](https://github.com/chaseBrow/CSE-40535-Vision/blob/main/example_image.jpg?raw=true)
