#se as pastas ja existirem, comentar as linhas que contem o comando "os.mkdir"

import cv2
import numpy as np
from skimage import io
import os


def global_contrast_normalization(patch,id_patch,id_img):
    media = np.mean(patch)
    desvioPadrao = np.std(patch)
    patch = np.subtract(patch,media)    
    patch = np.divide(patch,desvioPadrao)

    zca_whitening_matrix(patch,id_patch,id_img)
    
    cv2.imwrite("patchesGCN/img"+id_img+"_patch_"+id_patch+".png",patch)
    
def zca_whitening_matrix(patch,id_patch,id_img):
    sigma = np.cov(patch, rowvar=True)
    U,S,V = np.linalg.svd(sigma)
    epsilon = 1e-5
    zcaMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))

    cv2.imwrite("patchesZCA/img"+id_img+"_patch_"+id_patch+".png",patch)


def make_patches(img_mask,img_test,id_img,arq):
    cont = 0
    k = 0
    for x in range(0,img_mask.shape[1],27):
        for y in range(0,img_mask.shape[0],27):
            for x2 in range(x,x+27):
                for  y2 in range(y,y+27):
                    if x2< img_mask.shape[1] and y2< img_mask.shape[0]:
                        if img_mask[y2][x2] == 255:
                            cont += 1
            if cont == 729:
                patch = img_test[y:y+27, x:x+27]
                id_patch = str(k)
                
                arq.write(id_img+" "+id_patch+" "+str(x)+" "+str(y)+"\n")
                
                global_contrast_normalization(patch,id_patch,id_img)
                
                #cv2.imwrite("patches/img"+id_img+"_patch_"+id_patch+".png",patch)
                cv2.imwrite("patchesRetinex/img"+id_img+"_patch_"+id_patch+".png",patch)
                #print (x,y,x2,y2)
                k += 1
            cont = 0




dir = './patches'
os.mkdir(dir)

dir = './patchesGCN'
os.mkdir(dir)

dir = './patchesZCA'
os.mkdir(dir)

dir = "./patchesRetinex"
os.mkdir(dir)

arquivo = open('patches.txt','w')

for id in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']:
    img_mask = io.imread("mask/"+id+"_test_mask.gif")
    #img_test = cv2.imread("canalVerde/"+id+"_canalVerde.png",-1)
    img_test = cv2.imread("Retinex/"+id+"_retinex.png")
    make_patches(img_mask,img_test,id,arquivo)
arquivo.close()
                    
