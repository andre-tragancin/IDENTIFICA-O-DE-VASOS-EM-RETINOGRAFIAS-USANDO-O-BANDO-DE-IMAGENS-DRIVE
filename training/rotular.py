import cv2
import numpy as np
from skimage import io


def rotular(img,id_img,id_patch,x,y,arqW):
    cont = 0
    x_ = int(x)
    y_ = int(y)
    for x2 in range(x_,x_+28):
        for  y2 in range(y_,y_+28):
            if x2< img.shape[1] and y2< img.shape[0]:
                if img[y2][x2] >= 0.9:
                    cont += 1
    if cont >= 729*0.10:                #se contem no minimo 20% de vaso
        arqW.write('img'+id_img+'_patch_'+id_patch+'   \t TEM_VASOS \n')
    else:
        arqW.write('img'+id_img+'_patch_'+id_patch+'   \t NAO_TEM_VASOS \n')
    cont = 0
        


arquivoR = open('patches.txt','r')
arquivoW = open('rotularPatches.txt','w')
patches = arquivoR.readlines()

for linha in patches:
    
    info = linha.split()
    id_img = info[0]
    id_patch = info[1]
    x_img = info[2]
    y_img = info[3]

    img_manual = io.imread("1st_manual/"+id_img+"_manual1.gif",True)  #True -> deixa a imagem em tons de cinza
    rotular(img_manual,id_img,id_patch,x_img,y_img,arquivoW)

arquivoR.close()
arquivoW.close()
