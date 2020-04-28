from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import cv2
import numpy as np

radius = 1
n_points = radius * 8

matriz = [[8,2,7],
          [2,7,8],
          [7,7,1]]

arqRotulos = open("rotularPatches.txt","r")
texto = arqRotulos.readlines()
patch = texto[0]
patch = patch.split()
print (patch[0])
#image = cv2.imread('patchesZCA/'+patch[0]+'.png',-1)
image = cv2.imread('patchesZCA\img01_patch_0.png',-1)


lbp = local_binary_pattern(image, n_points, radius,method='uniform')

lbp2 = local_binary_pattern(matriz, n_points, radius, method='uniform')

print (lbp2)

(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, n_points + 3),
			range=(0, n_points + 2))
 
# normalize the histogram
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

for i in range(0,len(hist)):
    print (hist[i])

#print(len(lbp))
#print ("LPB=",lbp)
#print(list(map(str,lbp)))
string = ''
for i in range(0,len(hist)):
    converte = str(hist[i])
    string = string+converte+" "

#print (string)        
#print (str(hist))

#cv2.imwrite("ble4.png",lbp)
#cv2.imwrite("ble5.png",hist)

'''
import cv2
from skimage import io
import numpy as np


img = cv2.imread("images/01_test.tif")
img_manual = io.imread("2nd_manual/01_manual2.gif",True)
patch = img_manual[270:270+27, 27:27+27]
print (patch)
cv2.rectangle(img_manual,(27,270),(27+27,270+27),(0,255,0))

(b,g,r) = cv2.split(img)

b_media = np.mean(b)
g_media = np.mean(g)
r_media = np.mean(r)

b_desvio = np.std(b)
g_desvio = np.std(g)
r_desvio= np.std(r)

#np.subtract(a,b) == a-b
b = np.subtract(b, b_media)
g = np.subtract(g, g_media)
r = np.subtract(r, r_media)

#np.divide(a,b) == a/b
b = np.divide(b, b_desvio)
g = np.divide(g, g_desvio)
r = np.divide(r, r_desvio)

img = cv2.merge((b,g,r))

img_media = np.mean(img)
img_desvio = np.std(img)

img = np.subtract(img, img_media)
img = np.divide(img, img_desvio)



io.imsave("patchtest.png",patch)
io.imsave("imgtest.png",img_manual)
#cv2.imwrite("patchtest.png",patch)
#cv2.imwrite("imgtest.png",img_manual)
#cv2.imshow("img",img_manual)
#cv2.imshow("patch",patch)
#cv2.imshow("vermelho",r)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
