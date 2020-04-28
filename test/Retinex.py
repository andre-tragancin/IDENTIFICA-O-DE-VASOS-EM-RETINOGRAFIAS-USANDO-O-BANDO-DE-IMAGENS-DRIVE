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

for id in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']:
    img = cv2.imread("canalVerde/"+id+"_canalVerde.png")
    retinex = bioinspired_Retina(img)
    cv2.imwrite("retinex/"+id+"_retinex.png", retinex)

