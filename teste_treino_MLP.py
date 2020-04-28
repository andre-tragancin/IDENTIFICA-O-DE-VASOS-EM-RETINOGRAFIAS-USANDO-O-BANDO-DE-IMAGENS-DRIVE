from sklearn.svm import LinearSVC
import cv2
from skimage import io
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

def desenhaLinhas(img):
    for x in range(0,img.shape[0],27):
        cv2.line(img,(0,x+27),(img.shape[1],x+27),(255,255,255),1)
        
    for y in range(0,img.shape[1],27):
        cv2.line(img,(y+27,0),(y+27,img.shape[0]),(255,255,255),1)



arqLBPTeste = open("test/LBP_teste_Retinex_U.txt","r")
arqLBPTreino = open("training/LBP_treino_Retinex_U.txt","r")
textoTreino = arqLBPTreino.readlines()
textoTeste = arqLBPTeste.readlines()

x = []
y = []

for linha in textoTreino:
    info = linha.split()
    dados = list(map(float,info[0:729]))
    rotulo = info[729]
    x.append(dados)
    y.append(rotulo)

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20 , random_state=21)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20)
#x_train, x_test, y_train, y_test = train_test_split(x,y)


#PCA
#descritores para imagens de retine
#eye fundus images deescriptosr

#clf = MLPClassifier()
#clf = MLPClassifier( hidden_layer_sizes = ( 100, 100, 100), max_iter=200, activation = 'identity', solver = 'sgd', random_state=21)
clf = MLPClassifier( hidden_layer_sizes = ( 100, 100, 100), max_iter=200, activation = 'identity', solver = 'sgd')
#clf = MLPClassifier(activation = 'identity', solver = 'lbfgs')
clf.fit(x_train, y_train)

#print (x_train[0])

y_pred = clf.predict(x_test)

#print(clf.predict([[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.9999999000000099 ,0.0 ,0.0]]))

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
        predicao = clf.predict([dataTeste])
        #print(predicao)
        if (predicao == 'TEM_VASOS'):
            #print(k,"= patch",info[1])
            cv2.circle(imgTransformada,(meioX,meioY),3,(0,255,0,0.1),1)
            k += 1
        else:
            cv2.circle(imgTransformada,(meioX,meioY),3,(0,0,255,0.1),1)
    cont += 1

arqLBPTeste.close()
arqLBPTreino.close()
arqPatches.close()
            
cv2.imshow('img',imgTransformada)
cv2.waitKey(0)
cv2.destroyAllWindows()
