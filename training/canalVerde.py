import cv2
import os

def green_channel(img):
    b,g,r = cv2.split(img)
    return g

dir = './canalVerde'
os.mkdir(dir)

for id in ['21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40']:
    img = cv2.imread("images/"+id+"_test.tif")
    canalVerde = green_channel(img)
    cv2.imwrite("canalVerde/"+id+"_canalVerde.png",canalVerde)
