from sklearn.svm import LinearSVC
import cv2
from skimage import io
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

'''
def plot_learning_curve(estimator, title, x1, y1, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x1, y1, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
'''

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
    #print(len(info))
    dados = list(map(float,info[0:729]))
    rotulo = info[729]
    x.append(dados)
    y.append(rotulo)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state = 21)


#scaler = StandardScaler()

#scaler.fit_transform(x_train)

#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

pca = PCA(n_components=1)
pca.fit(x_train)

x_train = pca.transform(x_train)
x_test =  pca.transform(x_test)

clf = MLPClassifier( hidden_layer_sizes = ( 100, 100, 100), max_iter=500, activation = 'relu', solver = 'lbfgs', random_state = 21)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)


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
        dataTeste2 = pca.transform([dataTeste])
        #print(dataTeste)
        #print(dataTeste2)
        predicao = clf.predict(dataTeste2)
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
cv2.imwrite('Figura.png', imgTransformada)
cv2.waitKey(0)
cv2.destroyAllWindows()
