import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import cv2, time




#detection de faace
def dec_face(img, gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #parcours de l'image pur les face detecté
    for (x,y,w,h) in faces:
        faces = img[y:y + h, x:x + w]
        # cv2.imwrite('face.jpg', faces)
    return faces

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
#lecture de lim
img = cv2.imread("mosa1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

model_image = dec_face(img, gray)
cv2.imshow('img',model_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#lecture de l'image enrengitré men qbel et resized it
model_image = cv2.resize(model_image,(128,128),interpolation = cv2.INTER_AREA)


##############################################################################################

#loding image requete
path_name = ["rih1.jpg",  "rih2.jpeg", "pro41.jpg"]
faces = []

for i in path_name:
    img = cv2.imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = dec_face(img, gray)
    face = cv2.resize(face,(128,128),interpolation = cv2.INTER_AREA)
    faces.append(face)

for i in faces:
    # cv2.imshow('img',i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



###############################################LBP#############################################


#model image LBP
model_img_lbp = np.zeros((128,128))
model_gray = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)

for i in range(2, 127):
    for j in range(2, 127):
        # print(f[i][j])
        model_img_lbp[i-1][j-1] = lbp(model_gray,i,j)
cv2.imshow('img',model_img_lbp)
cv2.waitKey(0)
cv2.destroyAllWindows()

bloc_size_r = 8
bloc_size_c = 8
regions = []
region_model = create_regions(model_img_lbp,bloc_size_c,bloc_size_r)
desc_model = get_desc (region_model)

#requete image lbp
img_lbp = np.zeros((128,128))
list_lbp = []
img_mse = []
image_index = 0
 
for f in faces:
    fg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    for i in range(2, 127):
        for j in range(2, 127):
            # print(f[i][j])
            img_lbp[i-1][j-1] = lbp(fg,i,j)
    #defiition des regions de 8*8 d'une image lbp
    regions = create_regions(img_lbp,bloc_size_c,bloc_size_r)
    #defiition du descripteur a partir des region lbp
    desc = get_desc (regions)
    #calcule du mse
    mse = mean_squared_error(desc_model,desc)
    print("Le MSE entre l'image model et l'image numero " + str(image_index) + " est : " + str(mse) )
    img_mse.append((mse, image_index))
    image_index = image_index + 1 

match = sorted(img_mse)
print ("smallest MSE nigg : " + str(match[0][0]))
img_match = faces[match[0][1]]
plt.imshow(img_match)
plt.show()


##########################################################################################







