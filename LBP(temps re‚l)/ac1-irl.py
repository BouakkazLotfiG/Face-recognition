import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import cv2, time


###########################################################################
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
        cv2.imshow('img', gray)
        return []

def create_regions(test_image,bloc_size_r,bloc_size_c):
    regions = []
    for r in range(0,test_image.shape[0], bloc_size_r):
        for c in range(0,test_image.shape[1], bloc_size_c):
            window = test_image[r:r+bloc_size_r,c:c+bloc_size_c]
            regions.append(window)
    return np.array(regions)

def get_desc(regions):
    descriptor = []
    hist = []
    for r in regions :
        temp = r.ravel().tolist() 
        for i in range(256):
            hist.append(temp.count(i))
    descriptor.extend(hist)
    return descriptor

def lbp(M,i_ref,j_ref):
    ref_value = M[i_ref][j_ref]
    # print(ref_value)

    bin_val = ""
    #0
    if M[i_ref-1][j_ref-1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #1
    if M[i_ref-1][j_ref] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #2
    if M[i_ref-1][j_ref+1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #3
    if M[i_ref][j_ref+1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #4
    if M[i_ref+1][j_ref+1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #5
    if M[i_ref+1][j_ref] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #6
    if M[i_ref+1][j_ref-1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    #7 
    if M[i_ref][j_ref-1] >= ref_value :
        bin_val += "1"
    else :
        bin_val += "0"
    dec_val = int(bin_val,2) 
    #print(dec_val)
    return dec_val
##############################################################################################


img = cv2.imread("mosa1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

model_face = dec_face(gray)

cv2.imshow('img',model_face)
cv2.waitKey(0)

#lecture de l'image enrengitré men qbel et resized it
model_face = cv2.resize(model_face,(128,128),interpolation = cv2.INTER_AREA)


#model image LBP
model_face_lbp = np.zeros((128,128))

for i in range(2, 127):
    for j in range(2, 127):
        # print(f[i][j])
        model_face_lbp[i-1][j-1] = lbp(model_face,i,j)


bloc_size_r = 8
bloc_size_c = 8
regions = []
region_model = create_regions(model_face_lbp,bloc_size_c,bloc_size_r)
desc_model = get_desc (region_model)
# print(desc_model)
########################################################################################""

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_lbp = []

while True:
    _, img = video.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # faces = face_cascade.detectMultiScale(gray, 1.4, 5)

    faces = dec_face(gray)
    
    # print(faces)
    if (len(faces) > 0):
        face_lbp = np.zeros((128,128))
        faces = cv2.resize(faces,(128,128),interpolation = cv2.INTER_AREA)
        for i in range(2, 127):
            for j in range(2, 127):
                # print(f[i][j])
                face_lbp[i-1][j-1] = lbp(faces,i,j)

        regions = create_regions(face_lbp,bloc_size_c,bloc_size_r)
        #defiition du descripteur a partir des region lbp
        desc = get_desc (regions)
        #calcule du mse
        mse = mean_squared_error(desc_model,desc)

        if (mse <= 1.3):
            cv2.putText(gray, "lotfi", (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('img', gray)  
            print(mse)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break 

video.release()







##############################################################################################






 
# for f in faces:
#     fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
#     for i in range(2, 127):
#         for j in range(2, 127):
#             # print(f[i][j])
#             img_lbp[i-1][j-1] = lbp(fg,i,j)
#     #defiition des regions de 8*8 d'une image lbp
#     regions = create_regions(img_lbp,bloc_size_c,bloc_size_r)
#     #defiition du descripteur a partir des region lbp
#     desc = get_desc (regions)
#     #calcule du mse
#     mse = mean_squared_error(desc_model,desc)
#     print("Le MSE entre l'image model et l'image numero " + str(image_index) + " est : " + str(mse) )
#     img_mse.append((mse, image_index))
#     image_index = image_index + 1 

# match = sorted(img_mse)
# print ("smallest MSE nigg : " + str(match[0][0]))
# img_match = faces[match[0][1]]
# plt.imshow(img_match)
# plt.show()


##########################################################################################







