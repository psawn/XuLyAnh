import cv2
import numpy as np

img = cv2.imread('E:\\superman.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(gray, -1, kernelx)
img_prewitty = cv2.filter2D(gray, -1, kernely)

prewitt = img_prewittx + img_prewitty

edges = cv2.bitwise_not(prewitt)

color = cv2.bilateralFilter(img, d=2, sigmaColor=200,sigmaSpace=200)

edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

cartoon = cv2.addWeighted(edges,0.5,color,1,0)

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

img_1 = color_quantization(cartoon, 4)

cv2.imshow("Original",img)
cv2.imshow("Prewitt", prewitt)
cv2.imshow("cartoon", cartoon)
cv2.imshow("img_1", img_1)

cv2.imwrite("E:\Python\img\Original.png",img)
cv2.imwrite("E:\Python\img\Prewitt.png",prewitt)
cv2.imwrite("E:\Python\img\cartoon.png",cartoon)
cv2.imwrite("E:\Python\img\img_1.png",img_1)


cv2.waitKey(0)
cv2.destroyAllWindows()

