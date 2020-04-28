import cv2
import os

def green_channel(img):
    b,g,r = cv2.split(img)
    return g

dir = './canalVerde'
os.mkdir(dir)

for id in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']:
    img = cv2.imread("images/"+id+"_test.tif")
    canalVerde = green_channel(img)
    cv2.imwrite("canalVerde/"+id+"_canalVerde.png",canalVerde)
