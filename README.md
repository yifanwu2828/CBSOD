# Color-Based-Segmentation-and-Object-Detection
An object detection algorithm based on color segmentation with a probabilistic pixel color classifier and deterministic bounding box algorithm with object shape detection. If there exists a blue recycling bin, a bounding box will be drawn around the specific blue bin  


Object detection has applications in many areas of computer vision and image processing. Every object has its own special features such as size, dimension and color. Our object detection algorithm combined color segmentation with a pixel color classifier and bounding box algorithm with object dimension detection.


A machine learning approach was used to classify and extract blue color features. Three one-vs-all binary logistic regression models were developed to distinguish among red, green, and blue pixels. In order to improve the accuracy, the binary logistic regression model was retrained with additional hand-labeled pixel color data among bin-blue, non-bin-blue, black, red, yellow and green. A sufficient amount of pixel color data allowed the classifier to detect bin-blue region in an image. A Region of Interests  classification algorithm was developed to compare the similarity between the blue region of interest and blue recycling bin. This simple algorithm helped to detect the existence and location of blue recycling bin in the image.


![plot](/Results/0009.png)
![plot](/Results/0025.png)
