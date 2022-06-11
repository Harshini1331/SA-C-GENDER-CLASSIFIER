# SA-C-GENDER-CLASSIFIER
# Algorithm
1. Using the command pip, install deepface to install the DeepFace library.
2. Using the DeepFace library, we can predict the gender of a person.
3. Import the necessary packages like Deepface, cv2 and Matplot library.
4. Load and display the image which we have imported.
5. Pass the image to DeepFace library and analyze the image to predict gender of a person.
6. Pass the image to DeepFace library and analyze the image to predict emotions of a person.
7. This prediction is stored in result variable and print the prediction using this algorithm.

## Program:
```
/*
Program to implement 
Developed by   : Harshini M
RegisterNumber :  212220230022
*/
```
```python
pip install deepface

#import the packages
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

#Read the image
img=cv2.imread('img1.jpg')
plt.imshow(img[:,:,::-1])
plt.show()

#Analyze the gender
result = DeepFace.analyze(img,actions=['gender'])
print("Gender : ",result['gender'])

#Analyze the emotions
result2 = DeepFace.analyze(img,actions=['emotion'])
print("Emotions : ",result2['emotion'])

#Read the image
img1=cv2.imread('img2.jpg')
plt.imshow(img1[:,:,::-1])
plt.show()

#Analyze the gender
result3 = DeepFace.analyze(img1,actions=['gender'])
print("Gender : ",result3['gender'])

#Analyze the emotions
result4 = DeepFace.analyze(img1,actions=['emotion'])
print("Emotions : ",result4['emotion'])

```

## OUTPUT:
<img width="244" alt="image" src="https://user-images.githubusercontent.com/75235554/173192963-fa61afdd-aa3c-4f5c-b248-e10839edd372.png">
<img width="160" alt="image" src="https://user-images.githubusercontent.com/75235554/173192967-7e74e497-000a-4c48-bf65-8854e0d2387e.png">
<img width="723" alt="image" src="https://user-images.githubusercontent.com/75235554/173192979-ccf25195-5d82-41b3-8da4-fc59247328da.png">
<img width="225" alt="image" src="https://user-images.githubusercontent.com/75235554/173192985-1b938b18-d2b6-4d82-bf9c-1549981f7c7b.png">
<img width="146" alt="image" src="https://user-images.githubusercontent.com/75235554/173192995-e5aeadbc-97bb-406f-bf62-4b02379c2df3.png">
<img width="721" alt="image" src="https://user-images.githubusercontent.com/75235554/173193009-baa3c21f-0e80-4422-8873-93d6bd7ba081.png">



