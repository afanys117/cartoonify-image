import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_file(file_name):
    image = cv.imread(file_name)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.axis("off")
    return image

def edge_mask(image, line_size, blur_value):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray_blur = cv.medianBlur(gray, blur_value)

    edges = cv.adaptiveThreshold(
        gray_blur,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        line_size,
        blur_value
    )

    return edges


def color_quantization(image, k):
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)

    return result

def bilateral_filter(image):
    return cv.bilateralFilter(image_q, d=7, sigmaColor=200, sigmaSpace=200)

def cartoon(blurred, edges):
    return cv.bitwise_and(blurred, blurred, mask=edges)
    

image = read_file("image.png")
edges = edge_mask(image, 7, 5)

image_q = color_quantization(image, k=7)

blurred = bilateral_filter(image_q)
cartoon_image = cartoon(blurred, edges)
image = Image.fromarray(cartoon_image)
image.save("cartoon_image.png")
plt.imshow(cartoon_image)
plt.show()
