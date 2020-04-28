from sklearn.svm import LinearSVC
import cv2
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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


#print(label[0:10])
# Binarize the output
label = label_binarize(label, classes=["TEM_VASOS", "NAO_TEM_VASOS"])
n_classes = label.shape[1]

#print(label)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state = 21)

#pca transform
pca = PCA(n_components=1)
pca.fit(x_train)

x_train = pca.transform(x_train)
x_test =  pca.transform(x_test)

#train a Linear SVM on the data

model = LinearSVC()
#model.fit(x_train, y_train)
y_score = model.fit(x_train, y_train).decision_function(x_test)

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
        dataTeste2 = pca.transform([dataTeste])
        predicao = model.predict(dataTeste2)
        
        if (predicao == 0):
            #print(info[1])
            cv2.circle(imgTransformada,(meioX,meioY),3,(0,255,0,0.1),1)
        else:
            cv2.circle(imgTransformada,(meioX,meioY),3,(0,0,255,0.1),1)
    cont += 1
'''
#curva de aprendizado
title = "Curva de aprendizado(Linear SVC)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = model
plot_learning_curve(estimator, title, x_test, y_test, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()

#curva ROC
#n_classes = 2 #vaso e nao_vaso

print(y_test)
print(y_score)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''
cv2.imshow('img',imgTransformada)
cv2.waitKey(0)
cv2.destroyAllWindows()
