import numpy as np
import cv2 as cv
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

template_filename_vector = "pattern/detail.svg"
input_filenames = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
threshold = 0.05
template_filename = "pattern/detail.png"

# svg to bitmap
drawing = svg2rlg(template_filename_vector)
renderPM.drawToFile(drawing, template_filename, fmt="PNG")
template_filename = "pattern/template.jpg"
template = cv.imread(template_filename, 0)

def edge_image(image):
    img = cv.medianBlur(image, 15)
    img = cv.Canny(img, 20, 40)
    return img

def find_circles(image):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, minDist=20, param1=10, param2=20, maxRadius=50)
    circles = circles[0]

    print(circles)

    max_radius = np.max(list(map(lambda c: c[2], circles)))
    
    circles = list(filter(lambda c: abs(c[2] - max_radius) < 10, circles))
    return circles


def dist_matrix_determinant(circles):
    cn = len(circles)

    dist_matrix = np.zeros((cn, cn))

    for i in range(cn):
        for j in range(cn):
            dist_matrix[i, j] = np.sqrt(
                (circles[i][0] - circles[j][0]) ** 2
                + (circles[i][1] - circles[j][1]) ** 2
            )

    dist_matrix = dist_matrix / np.max(dist_matrix)

    return np.linalg.det(dist_matrix)


def embedding_diff(e1, e2):
    return abs(abs(e1) - abs(e2))


original_embedding = dist_matrix_determinant(find_circles(template))

result = {
    0: "OK",
    1: "Item is not polished or too much input noise",
    2: "Two holes missed or incomplete",
    3: "One hole missed or incomplete",
    4: "Wrong holes geometry",
}


def check_defect(circles):
    if len(circles) < 2 or len(circles) > 4:
        return result[1]
    if len(circles) < 3:
        return result[2]
    if len(circles) < 4:
        return result[3]

    input_embedding = dist_matrix_determinant(circles)

    difference = embedding_diff(original_embedding, input_embedding)

    if difference > threshold:
        return result[4]

    return result[0]

def draw_circles(file, image, c):
    for circle in c:
        cv.circle(image,(int(circle[0]), int(circle[1])), int(circle[2]), (255,0,17), -1)
    cv.imwrite('output/'+file, image)

for filename in input_filenames:
    img = cv.imread('input/'+filename, 0)
    edged = edge_image(img)
    circles = find_circles(edged)
    draw_circles(filename, edged, circles)
    result_text = check_defect(circles)
    print(result_text)

    # print to image
    text_start = (5, 30)
    rect = np.array(
        [
            [
                [text_start[0] - 10, text_start[1]],
                [text_start[0] + 500, text_start[1]],
                [text_start[0] + 500, text_start[1] - 30],
                [text_start[0] - 10, text_start[1] - 30],
            ]
        ],
        dtype=np.int32,
    )
    cv.fillPoly(img, rect, (255, 255, 255), cv.LINE_4)
    cv.putText(img, result_text, text_start, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
    cv.imwrite("output/result/"+filename, img)
