import numpy as np
import cv2 as cv
from scipy.signal import convolve2d
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

def transfor_to_greyscale(pic):
    pic = pic[::, ::, 0]*0.2126 + pic[::, ::, 1]*0.7152 + pic[::, ::, 2]*0.0722
    return pic

def get_gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def get_peaks(H, num_peaks, nhood_size_x = 20, nhood_size_y = 20):
    H1 = H.copy()
    indicies = []
    for i in range(num_peaks):
        H1_idx = np.unravel_index(np.argmax(H1), H1.shape)
        indicies.append((H1_idx[1], H1_idx[0]))
        x, y = H1_idx
        x_min = max(0, x-nhood_size_x)
        x_max = min(H1.shape[0], x+nhood_size_x)
        y_min = max(0, y-nhood_size_y)
        y_max = min(H1.shape[1], y+nhood_size_y)
        H1[x_min:x_max, y_min:y_max] = 0
    return indicies


def preproc(img):
    if len(img.shape)==3:
        img2 = transfor_to_greyscale(img)
    else: img2 = img.copy()
    img2= img2.astype(np.uint8)
    
    gaussian_kernel = get_gaussian_kernel(5)
    pic_g_blur = convolve2d(img2, gaussian_kernel, mode = 'same')
    pic_g_blur[0] = pic_g_blur[1]
    pic_g_blur[-1] = pic_g_blur[-2]
    pic_g_blur[::,0] = pic_g_blur[::,1]
    pic_g_blur[::,-1] = pic_g_blur[::,-2]

    sobel_filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_x = sobel_filter_y.T
    ##SOBEL
    def apply_sobel_filter(pic_x):
        x_diff = convolve2d(pic_x, sobel_filter_x, mode = 'same')
        y_diff = convolve2d(pic_x, sobel_filter_y, mode = 'same')

        grad = np.sqrt(x_diff**2 + y_diff**2)
        grad[0] = grad[1]
        grad[-1] = grad[-2]
        grad[::,0] = grad[::,1]
        grad[::,-1] = grad[::,-2]

        return x_diff, y_diff, grad

    x_diff, y_diff, pic_edge = apply_sobel_filter(pic_g_blur)
    
    
    gaussian_kernel = get_gaussian_kernel(10)
    pic_edge = convolve2d(pic_edge, gaussian_kernel, mode = 'same')
    pic_edge[0] = pic_edge[1]
    pic_edge[-1] = pic_edge[-2]
    pic_edge[::,0] = pic_edge[::,1]
    pic_edge[::,-1] = pic_edge[::,-2]
    pic_edge = pic_edge.astype(np.uint8)
    
    return pic_edge

img2 = cv.imread('input/DSC_4938.JPG')

img2 = preproc(img2)
cv.imwrite("output/img1.jpg", img2)

drawing = svg2rlg('detail.svg')
renderPM.drawToFile(drawing, "output/template.png", fmt="PNG")
template = cv.imread('output/template.png',0)

l1 = 0
l2 = 0.5
template = template[int((l1)*template.shape[0]):int((l2)*template.shape[0]), int((l1)*template.shape[0]):int((l2)*template.shape[0])]

# template =  cv.resize(template, (0,0), fx=0.7, fy=0.7) 
template = 255 - template
template = preproc(template)
w, h = template.shape[::-1]
cv.imwrite("output/template.jpg", template)

img = img2.copy()
cv.imwrite("output/img2.jpg", img)

# Apply template Matching
res = cv.matchTemplate(img, template ,cv.TM_CCORR_NORMED)
cv.imwrite("output/map.jpg", res)

peaks = get_peaks(res, 2, nhood_size_x = 100, nhood_size_y = 100)
img_z = np.zeros(img2.shape)
for p in peaks:
    x = p[1]
    y = p[0]
    div = 1
    img_z[x:x+int(w/div), y:y+int(h/div)] = \
                    img.max()/2
    
img_z = img_z+img

cv.imwrite("output/result.jpg", img_z)