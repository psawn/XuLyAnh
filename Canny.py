import cv2
import numpy as np

img = cv2.imread('E:\\superman.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(img,150,150)

kernel = np.ones((1,2),np.uint8)
erosion = cv2.erode(canny,kernel,iterations = 1)
dilation = cv2.dilate(canny,kernel,iterations = 2)

edges = cv2.bitwise_not(dilation)

color = cv2.bilateralFilter(img, d=9, sigmaColor=300,sigmaSpace=300)

cartoon = cv2.bitwise_and(color, color, mask=edges)

def color_quantization(img, k):
# Defining input data for clustering
  data = np.float32(img).reshape((-1, 3))
# Defining criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying cv2.kmeans function
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

img_1 = color_quantization(cartoon, 2)

cv2.imshow("Original",img)
cv2.imshow("Edges",edges)
cv2.imshow("Cartoon",cartoon)
cv2.imshow("Cartoon2",img_1)
cv2.imshow("canny",canny)


cv2.waitKey(0)
cv2.destroyAllWindows()
