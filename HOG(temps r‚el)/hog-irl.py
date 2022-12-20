import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 
from sklearn.metrics import mean_squared_error
from PIL import Image



#detection de faace
def dec_face(gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #parcours de l'image pur les face detecté
    if (len(faces) != 0):
        for (x,y,w,h) in faces:
            faces = gray[y:y + h, x:x + w]
            cv2.rectangle(gray, (x, y), (x+w, y+h), (255,0,0),2)
            cv2.imshow('img', gray)
        return faces
    else : 
        # cv2.imshow('img', gray)
        return []

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
            c = a[i][j] * a[i][j] + b[i][j] * b[i][j]
            row.append(round(math.sqrt(c), 3))
        res.append(row)
    return res

def dir(a, b):
    res = []
    for i in range(0, 255):
        row = []
        for j in range(0, 255):
            if (a[i][j] == 0 ) : 
                a[i][j] = 1
            c = math.degrees(math.atan(b[i][j]/a[i][j]))
            c = round(c, 3)
            row.append(c)
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

########################################################################################""

model = cv2.imread("mosa1.png")
gray = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

model_face = dec_face(gray)

resized = cv2.resize(model_face,(256,256),interpolation = cv2.INTER_AREA)
cv2.imshow('img',resized)
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
direction_list = dir(X, Y)
direction = np.asarray(direction_list)
# plt.imshow(direction,cmap='gray')
# plt.show()


#calcule histogramme et descripteur
bloc_size_r = 8
bloc_size_c = 8
region = create_regions(direction,bloc_size_c,bloc_size_r)
desc_model = get_desc (region)
# print(desc)


##########################################################################
desc_list = []
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
bloc_size_r = 8
bloc_size_c = 8

while True:
    _, img = video.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # faces = face_cascade.detectMultiScale(gray, 1.4, 5)

    faces = dec_face(gray)
    # cv2.imshow('img',faces)
    
    # print(faces)
    
    if (len(faces) > 0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img',gray)
        resized = cv2.resize(faces,(256,256),interpolation = cv2.INTER_AREA)
        
        #calcule de gradient X
        X = gradX(resized)
        # plt.imshow(X,cmap='gray')
        # plt.show()

        #calcule de gradient Y
        Y = gradY(resized)
        # plt.imshow(Y,cmap='gray')
        # plt.show()

        #matrice G
        mag = addM(X, Y)
        # plt.imshow(mag,cmap='gray')
        # plt.show()

        #direction
        direction_list = dir(X, Y)
        direction = np.asarray(direction_list)
        # plt.imshow(direction,cmap='gray')
        # plt.show()
        
        #calcule histogramme et descripteur
        region = create_regions(direction,bloc_size_c,bloc_size_r)
        desc   = get_desc (region)
        mse    = mean_squared_error(desc, desc_model)
        print(mse)

        if (mse <= 1):
            cv2.rectangle(img, (int(faces[0][0]), int(faces[0][1])), (int(faces[0][2]), int(faces[0][3])), (255,0,0),2)
            cv2.putText(gray, "moussa", (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('img', gray)
            

            
        # print(desc)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break 
video.release()


#####################################################################



