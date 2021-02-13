# Color Segmentation, Image Processing and Bounding Box Algo

## Image segmentation 
Image segmentation segments the 3-D color space into a
set of volumes associated with different colors. A million
hand-labeled training pixel data with both positive examples
and negative examples is collected with respect to 6 color
class categories (bin-blue, non=bin-blue, black, red, yellow,
green). Each pixel is a 3-D vector x = (R, G, B). The model 
takes entire original RGB color images as input which is a
3 dimensional matrix M ∈ Z m×n×3
, and uses probabilistic
0
pixel color classifier to classified all pixels in the image. The
output is a binary image mask, M ∈ Z 0 m×n , which indicates
region of interests based on the bin-blueness. Each entry of
matrix, M i,j ∈ {0, 1} is 1 indicates the pixel has bin-blue
color and 0 otherwise.


![plot](/Results/img_msk_valid1.png)
![plot](/Results/img_msk_valid2.png)
![plot](/Results/img_msk_valid3.png)


## Image Processing
A large portion of blue recycling bin is recognizable in the
image mask, but the model is not perfect due to some low
bias and high variance. It misclassifies some pixel x j ∈ R 3
with similar value in the RGB color space. In order to filter
out noisy objects in an image mask, morphological operations
(e.g., dilation or erosion) was performed on the output binary
image mask.


![plot](/Results/img_process1.png)
![plot](/Results/img_process2.png)
![plot](/Results/img_process3.png)


## ROIs and ounding Box Algo
Region of Interests classification evaluates similarity be-
tween region of interests by size, dimension orientation etc.
Figure 2 shows the dimension of three blue recycling bins
with different sizes respectively [1]. In each size of blue bin,
H denoted the height, and, W denoted the width of a regular
recycling bin. Similarity is largely weighted on shape ratio
of blue bin, H/W ∈ R, Other criteria such as ratio between
region area and bounding box area, ratio between region area
and entire area of image, and orientations of the bin were
evaluated as well. The input is a prepossessed image mask,
M ∈ Z m×n
, and output is a list of bounding box coordinates[
0
(x 1 , y 1 ), (x 2 , y 2 ) ] corresponding to coordinates of upper-left
corner and lower-right corner of output bounding box. Figure
1 displays a sample result of blue-bin detector with a red
rectangular bounding box around a blue bin.


![plot](/Results/myplot1.png)
![plot](/Results/myplot2.png)
![plot](/Results/myplot4.png)
