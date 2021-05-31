import numpy as np
import cv2

img = cv2.imread('./input/DSC_4944.JPG')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blured = cv2.GaussianBlur(imgGry, (25, 25), 0)

 
grad_x = cv2.Sobel(blured, cv2.CV_16S, 1, 0, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(blured, cv2.CV_16S, 0, 1, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)   
blured_sobel = cv2.GaussianBlur(sobel, (75, 75), 0)
ret , thrash = cv2.threshold(blured_sobel, 120, 250, cv2.CHAIN_APPROX_NONE)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.imwrite('./output/rect-sobel.png', blured_sobel)
cv2.imwrite('./output/rect-tresh.png', thrash)


for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        cv2.putText( img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
    elif len(approx) == 4 :
        x, y , w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio < 1.05:
            cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        else:
            cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    elif len(approx) == 5 :
        cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 10 :
        cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    else:
        cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

cv2.imwrite('./output/rect.png', img)