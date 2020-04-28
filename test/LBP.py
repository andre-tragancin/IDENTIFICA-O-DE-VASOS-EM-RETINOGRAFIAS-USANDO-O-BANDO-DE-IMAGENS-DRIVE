from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import cv2
import numpy as np


def local_binary_patterns(patch, radius, n_points):
    lbp = local_binary_pattern(patch, n_points, radius, method='uniform')

    #(hist, _) = np.histogram(lbp.ravel(),
#			bins=np.arange(0, n_points + 3),
#			range=(0, n_points + 2))
 
    # normalize the histogram
    #hist = hist.astype("float")
    #hist /= (hist.sum() + 1e-7)
    return lbp

def lbpToStr(lbp):
    string = ''
    for i in range(0,len(lbp)):
        toStr = str(lbp[i])
        toStr = toStr.split("\n")
        toStr = toStr[0][1:72]+toStr[1][0:9]
        string = string+toStr+" "
    return string


radius = 1
n_points = 8 * radius
arqRotulos = open("rotularPatches.txt","r")
texto = arqRotulos.readlines()
#arqLBP = open("LBP_rotulo.txt","w")
arqLBP = open("LBP_teste_GCN_U.txt","w")
for linha in texto:
    info = linha.split()
    nome_patch = info[0]
    #rotulo_patch = info[1]
    patch = cv2.imread("patchesGCN/"+nome_patch+".png",0)
    lbp = local_binary_patterns(patch,radius,n_points)
    string = lbpToStr(lbp)
    arqLBP.write(string+"\n")
    #arqLBP.write(string+"\t"+rotulo_patch+"\n")
    
arqRotulos.close()
arqLBP.close()
