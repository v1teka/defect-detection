import numpy as np
import cv2 as cv
import imutils

image = cv.imread("input/photo.jpg")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (15, 15), 0)
cv.imwrite("output/gray.jpg", gray)

 
grad_x = cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(gray, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)   
cv.imwrite("output/sobel.jpg", sobel)

laplaced = cv.Laplacian(sobel, cv.CV_8U, ksize=5)
cv.imwrite("output/laplace.jpg", laplaced)

edged = cv.Canny(sobel, 10, 250)
cv.imwrite("output/edged.jpg", edged)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
cv.imwrite("output/closed.jpg", closed)

cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
total = 0

initial_angle = None

for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.1 * peri, True)

    if (len(approx) in [4]):
        cv.drawContours(image, [approx], -1, (0, 255, 0), 4)

        rows,cols = image.shape[:2]
        [vx,vy,x,y] = cv.fitLine(c, cv.DIST_L2, 0, 0.0001, 0.0001)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)

        line_start = (cols-1,righty)
        line_end = (0,lefty)
        cv.line(image, line_start, line_end, (0,255,0), 2)
        
        angle_pi = (righty - lefty)/(cols-1)
        angle = int(np.arctan(angle_pi)*180/np.pi)
        
        text_start = (int(x[0]),int(y[0]))
        rect = np.array( [[[text_start[0]-10,text_start[1]],[text_start[0]+50,text_start[1]],[text_start[0]+50,text_start[1]-30],[text_start[0]-10,text_start[1]-30]]], dtype=np.int32 )
        cv.fillPoly(image, rect, (255, 255, 255), cv.LINE_4)
        cv.putText(image, str(angle), text_start, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))

cv.imwrite("output/output.jpg", image)