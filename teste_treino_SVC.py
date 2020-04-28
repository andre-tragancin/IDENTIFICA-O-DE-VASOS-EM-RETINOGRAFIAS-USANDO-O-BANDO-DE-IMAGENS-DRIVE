from sklearn.svm import LinearSVC
import cv2
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA

def desenhaLinhas(img):
    for x in range(0,img.shape[0],27):
        cv2.line(img,(0,x+27),(img.shape[1],x+27),(255,255,255),1)
        
    for y in range(0,img.shape[1],27):
        cv2.line(img,(y+27,0),(y+27,img.shape[0]),(255,255,255),1)
            

#dir = './resultadosSVC'
#os.mkdir(dir)

arqLBPTeste = open("test/LBP_teste_Retinex_U.txt","r")
arqLBPTreino = open("training/LBP_treino_Retinex_U.txt","r")
textoTreino = arqLBPTreino.readlines()
textoTeste = arqLBPTeste.readlines()

data = []
label = []

for linha in textoTreino:
    info = linha.split()
    dados = list(map(float,info[0:729]))
    data.append(dados)
    label.append(info[729])

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state = 21)


#train a Linear SVM on the data

model = LinearSVC(max_iter = 15000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print (accuracy_score(y_test, y_pred))
print (confusion_matrix(y_test, y_pred))

id = '01'
img = cv2.imread("test/Retinex/"+id+"_retinex.png")
imgTransformada = img
desenhaLinhas(imgTransformada)
arqPatches = open("test/patches.txt","r")
textoPatches = arqPatches.readlines()

cont = 0
k=1
for linhaPatches in textoPatches:
    info = linhaPatches.split()
    id_img = info[0]
    x = info[2]
    y = info[3]
    x = int(x)
    y = int(y)
    meioX = int(x+13.5)
    meioY = int(y+13.5)
    if id_img == id:
        dataTeste = textoTeste[cont].split()[0:729]
        dataTeste = list(map(float,dataTeste))
        predicao = model.predict([dataTeste])
        
        if (predicao == 'TEM_VASOS'):
            #print(info[1])
            cv2.circle(imgTransformada,(meioX,meioY),3,(0,255,0,0.1),1)
        else:
            cv2.circle(imgTransformada,(meioX,meioY),3,(0,0,255,0.1),1)
    cont += 1
            
cv2.imshow('img',imgTransformada)
cv2.waitKey(0)
cv2.destroyAllWindows()
