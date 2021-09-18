# CSE-40535-Vision

Crystal Colon: **ccolon2**  
Chase Brown: **cbrown54**

https://www.kaggle.com/grassknoted/asl-alphabet

Description:
A.)The dataset contains 87,000 200x200 images of the ASL alphabet. There are 29 classes, of which 26 are for the letters A-Z and 3 classes are for SPACE, DELETE, and NOTHING.

B)  
   The data is already nicely split into testing and training data, however, there is only one photo for each of the 29 classes in the test folder.  Meanwhile there are 3000 photos for each class in the training folder.  We will have to move a couple hundred photos from the training data into the testing data for each class and furthermore split the data testing data into two groups, validation and testing.
  
After looking at several images we believe key features we should focus on identifying are knuckles and finger nails.  depending on if you extend your fingers or scrunch your hand in a fist, your knuckles can be the darkest or the lightest area on your hand.  A large portion of the photos are taken with no foreground light and you can only see the silhouette of the hand.  As far as we are aware, the only option in this case would be to indentify shapes by the outline of the hand.  

Another idea 
