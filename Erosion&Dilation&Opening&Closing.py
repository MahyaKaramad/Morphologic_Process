import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# First Image 
original_img1 = cv.imread("Handwriting.jpg" , 0)

retval1,mask1 = cv.threshold(original_img1 , 150, 255 , cv.THRESH_BINARY)

# Erosion  (Make the text thinner)
kernel1 = np.ones((3,3) , np.uint8)
eroded_img = cv.erode(original_img1 , kernel1 , iterations=1)

# Dilation   
kernel2 = np.ones((3,3) , np.uint8)
dilated_img = cv.dilate(original_img1 , kernel2 , iterations=1)


#.................................................
# Second Image 

original_img2 = cv.imread("noise.jpg" , 0)
retval2,mask2 = cv.threshold(original_img2 , 150, 255 , cv.THRESH_BINARY)


#bluring
median_filtered = cv.medianBlur(original_img2, 5)

#opening
kernel4 = np.ones((5,5) , np.uint8)
opened_img = cv.morphologyEx (median_filtered , cv.MORPH_OPEN ,kernel4 )


#closing
kernel3 = np.ones((7,7) , np.uint8)
closed_img = cv.morphologyEx (opened_img , cv.MORPH_CLOSE ,kernel3 )


#plot1 
images = [original_img1, mask1, eroded_img, dilated_img]
titles = ["Original", "Mask", "Erosion", "Dilation"]

plt.figure(figsize=(10, 8))  # Set figure size
for i in range(len(images)):
    plt.subplot(2, 2, i + 1)  # Create a grid with 2 rows and 2 columns
    plt.imshow(images[i], cmap="gray")  # Display image in grayscale
    plt.title(titles[i])  # Add title to each subplot
    plt.axis("off")  # Turn off axes for cleaner display
plt.tight_layout()  # Adjust layout for better spacing



# plot2
images = [original_img2, mask2, closed_img, opened_img]
titles = ["Original", "Mask", "Closing", "Opening"]

plt.figure(figsize=(10, 8))  # Set figure size
for i in range(len(images)):
    plt.subplot(2, 2, i + 1)  # Create a grid with 2 rows and 2 columns
    plt.imshow(images[i], cmap="gray")  # Display image in grayscale
    plt.title(titles[i])  # Add title to each subplot
    plt.axis("off")  # Turn off axes for cleaner display
plt.tight_layout()  # Adjust layout for better spacing



plt.show()