import cv2# cv2 versao 4.0
import os

def bioinspired_Retina(img):
    retina = cv2.bioinspired_Retina.create((img.shape[1],img.shape[0]))
    retina.run(img)

    retinaOut_parvo=retina.getParvo()
    retinaOut_magno=retina.getMagno()

    return retinaOut_parvo

dir = "./Retinex"
os.mkdir(dir)

for id in ['21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40']:
    img = cv2.imread("canalVerde/"+id+"_canalVerde.png")
    retinex = bioinspired_Retina(img)
    cv2.imwrite("retinex/"+id+"_retinex.png", retinex)

