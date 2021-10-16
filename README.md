# CSE-40535-Vision

Crystal Colon: **ccolon2**  
Chase Brown: **cbrown54**

https://www.kaggle.com/grassknoted/asl-alphabet

Description:
A.)The dataset contains 87,000 200x200 images of the ASL alphabet. There are 29 classes, of which 26 are for the letters A-Z and 3 classes are for SPACE, DELETE, and NOTHING.

B)  
   The data is already nicely split into testing and training data, however, there is only one photo for each of the 29 classes in the test folder.  Meanwhile there are 3000 photos for each class in the training folder.  We will have to move a couple hundred photos from the training data into the testing data for each class and furthermore split the data testing data into two groups, validation and testing.
  
After looking at several images we believe key features we should focus on identifying are knuckles and finger nails.  depending on if you extend your fingers or scrunch your hand in a fist, your knuckles can be the darkest or the lightest area on your hand.  A large portion of the photos are taken with no foreground light and you can only see the silhouette of the hand.  As far as we are aware, the only option in this case would be to indentify shapes by the outline of the hand.  

Another idea is that your fingers will form straight lines when two or more fingers are aligned next to eachother.  We may be able to create a filter which identifies straight lines well.  

Our last piece of information we noticed is the importance of the thumb in ASL.  We believe it will help greatly if we can identify the position of the thumb.  We have noticed the finger nail of the thumb is generally perpendicular to the rest of your finger nails.  We are not sure of any strategies to help identify this feature, but if there are any, we believe it might be important.

**Image Loading**
We did not do any manual data collection. Each image comes from our datasource as 200x200. This is already a pretty good resolution for our future work, however, we did implement in our image reader a way to resize the images again. The reason we chose to implement this functionality is incase we attempt to implement some sort of a Convoluntional Neural Network as they do not need images with such high resolution and it will likely show diminishing returns above 64x64.

**PreProcessing**
