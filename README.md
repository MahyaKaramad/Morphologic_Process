# Morphological Operations and Noise Removal with OpenCV

This project demonstrates how to apply **image processing techniques** using OpenCV and Matplotlib. It covers morphological operations such as **erosion**, **dilation**, **opening**, and **closing**, along with **thresholding** and **median filtering** on two sample images:
1. **Handwriting Image**: To thin or thicken text using erosion and dilation.
2. **Noisy Image**: To reduce noise using median filtering and morphological operations like opening and closing.

---

## Features

1. **Morphological Operations**:
   - **Erosion**: Shrinks white regions to thin text or remove small noise.
   - **Dilation**: Expands white regions to thicken text or fill gaps.
   - **Opening**: Removes small white noise.
   - **Closing**: Fills small black gaps in white regions.

2. **Noise Reduction**:
   - **Median Filtering**: Effectively removes salt-and-pepper noise.

3. **Thresholding**:
   - Converts grayscale images into binary images for morphological processing.

4. **Visualization**:
   - Displays the original and processed images side by side using Matplotlib.

---

## Requirements

Make sure you have the following installed:
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

Install the required libraries using pip:
```bash
pip install opencv-python-headless numpy matplotlib
