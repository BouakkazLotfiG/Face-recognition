import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 
from sklearn.metrics import mean_squared_error



#detection de faace
def dec_face(gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.8, 4)
    #parcours de l'image pur les face detecté
    return faces

def gradX (img):
    gradX = []
    grad = []
    for i in img:
        k=1
        j=0
        while k <= 255:
            grad.append(int(i[k]) - int(i[j]))
            k = k + 1
            j = j + 1
        gradX.append(grad)
        grad = []
    array = np.array(gradX)
    return(array)

def gradY (img):
    gradY = []
    grad = []
    for i in img.T:
        k=1
        j=0
        while k <= 255:
            grad.append(int(i[k]) - int(i[j]))
            k = k + 1
            j = j + 1
        gradY.append(grad)
        grad = []
    array = np.array(gradY)
    return(array.T)

def addM(a, b):
    res = []
    for i in range(0, 255):
        row = []
        for j in range(0, 255):
            if (a[i][j] == 0 ) : 
                a[i][j] = 1
            c = a[i][j] * a[i][j] + b[i][j] * b[i][j]
            row.append(round(math.sqrt(c), 3))
        res.append(row)
    return res

def dir(a, b):
    res = []
    for i in range(0, 255):
        row = []
        for j in range(0, 255):
            c = math.degrees(math.atan(b[i][j]/a[i][j]))
            row.append(round(c, 3))
        res.append(row)
    return res

def create_regions(test_image,bloc_size_r,bloc_size_c):
    regions = []
    for r in range(0, test_image.shape[0], bloc_size_r):
        for c in range(0, test_image.shape[1], bloc_size_c):
            window = test_image[r:r+bloc_size_r,c:c+bloc_size_c]
            regions.append(window)
    regions = np.array(regions, dtype=object)
    return regions 

def get_desc(regions):
    descriptor = []
    hist = []
    for r in regions :
        temp = r.ravel().tolist() 
        for i in range(256):
            hist.append(temp.count(i))
    descriptor.extend(hist)
    return descriptor


##################################################################
img1 = cv2.imread("brad.jpg")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img1)
cv2.waitKey(0)
desc1 = []
faces = dec_face(gray1)
for (x,y,w,h) in faces:
    face = img1[y:y + h, x:x + w]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(face_gray,(256,256),interpolation = cv2.INTER_AREA)
    # cv2.imshow('img',resized)
    cv2.waitKey(0)

    #calcule de gradient X
    X = gradX(resized)
    plt.imshow(X,cmap='gray')
    # plt.show()

    #calcule de gradient Y
    Y = gradY(resized)
    plt.imshow(Y,cmap='gray')
    # plt.show()

    #matrice G
    mag = addM(X, Y)
    plt.imshow(mag,cmap='gray')
    # plt.show()

    #direction
    direction_array = dir(X, Y)
    direction = np.array(direction_array)
    plt.imshow(direction,cmap='gray')
    # plt.show()
    

    #calcule histogramme et descripteur
    bloc_size_r = 8
    bloc_size_c = 8
    region = create_regions(direction,bloc_size_c,bloc_size_r)
    desc = get_desc (region)
    desc1.append((desc, face))
    # print(desc)

#####################################################################""

img2 = cv2.imread("leoNbrad.jpg")
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img2)
cv2.waitKey(0)
desc2 = []
faces = dec_face(gray2)
for (x,y,w,h) in faces:
    face = img2[y:y + h, x:x + w]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(face_gray,(256,256),interpolation = cv2.INTER_AREA)
    # cv2.imshow('img',resized)
    cv2.waitKey(0)

    #calcule de gradient X
    X = gradX(resized)
    plt.imshow(X,cmap='gray')
    # plt.show()

    #calcule de gradient Y
    Y = gradY(resized)
    plt.imshow(Y,cmap='gray')
    # plt.show()

    #matrice G
    mag = addM(X, Y)
    plt.imshow(mag,cmap='gray')
    # plt.show()

    #direction
    direction_array = dir(X, Y)
    direction = np.array(direction_array)
    plt.imshow(direction,cmap='gray')
    # plt.show()
    

    #calcule histogramme et descripteur
    bloc_size_r = 8
    bloc_size_c = 8
    region = create_regions(direction,bloc_size_c,bloc_size_r)
    desc = get_desc (region)
    desc2.append((desc, face))
    # print(desc)

#####################################################################
mse1 = mean_squared_error(desc1[0][0],desc2[0][0])
mse2 = mean_squared_error(desc1[0][0],desc2[1][0])

print(mse1)
# print(mse2)
# print(mse3)
# print(mse4)

if (mse1 < mse2):
    cv2.imshow('img',desc2[0][1])
    cv2.waitKey(0)
else:
    cv2.imshow('img',desc2[1][1])
    cv2.waitKey(0)



cv2.destroyAllWindows()