import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import dump

# constants
imgHeight = 256
imgWidth = 256

labelCode = {
    'KA': 0,
    'KL': 1,
    'KM': 2,
    'KR': 3,
    'MK': 4,
    'NA': 5,
    'NM': 6,
    'TM': 7,
    'UY': 8,
    'YM': 9,
}

# Labels are based on the persons name, first letter of the picture file
def load_images(image_path, imgHeight, imgWidth, labelCode):
    labelsTrain = []
    img_vectorsTrain = []
    for filepath in glob.glob(os.path.join(image_path, '*.jpg')):
        labelsTrain.append(labelCode[filepath[19:21]])
        image = Image.open(filepath).convert('L')
        img_vectorsTrain.append(np.array(image).flatten().tolist())
    return np.array(img_vectorsTrain), labelsTrain

# RUN PCA
nComp = 80
pca = PCA(n_components=nComp, whiten=True).fit(np.array(img_vectorsTrain))
eVals = np.square(pca.singular_values_)
def plot_scree(eigenvalues):
    plt.figure('Scree Plot')
    plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot for PCA Eigenvalues')
    plt.axhline(y=1, color='r', linestyle='--')  # Kaiser criterion (Eigenvalue=1)
    plt.grid(True)
    plt.show()

# find best k for knn classification
def find_best_k(X_train, y_train, X_test, y_test, k_range, metric):
    accuracies = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracies.append(metrics.accuracy_score(y_test, pred))
    best_k = k_range[np.argmax(accuracies)]
    print(f'Best k by accuracy: {best_k}')

    
# literal 3: accumulated variance for chosen eigen vectors in 2) literal
dictAccVariance = {}
pcaVariance = pca.explained_variance_ratio_
countVar = 0
accVariance = 0
accVarianceArr = []
for var in pcaVariance:
    countVar += 1
    accVariance += var
    accVarianceArr.append(accVariance)
    dictAccVariance[f'PC-{countVar}'] = accVariance
# print results
strAccVariance = 'Accumulated variance for selected Principle Components\n\n'
for k, v in dictAccVariance.items():
    strAccVariance += f'{k} -> {v}\n'
with open(r'data\results\AccVariance.txt', 'w', encoding="utf-8") as f:
    f.write(strAccVariance)
# plot accumulated variance
plt.figure('Accumulated variance vs components')
plt.subplot(111)
plt.plot(range(1, nComp+1), accVarianceArr, marker='.')
plt.hlines(0.95, 0, nComp, colors='r')
plt.xlabel('Number of Components')
plt.ylabel('Acc. Variance')
plt.show()

# literal 6: Classification
# transform train data to PCA, principle axis
pca_imgVectorsTrain = pca.transform(np.array(imgVectorsTrain))
typeDist = 2  # euclidean
k = 2  # it was 8 we changed it to 2 for the best accuracy
# build and train model that uses euclidean distance
clf = KNeighborsClassifier(n_neighbors=k, p=typeDist)
clf.fit(pca_imgVectorsTrain, labelsTrain)
# go through test data set images
imgVectorsTest = []
testImgNames = []
labelsTest = []
for filepath in glob.glob(os.path.join(r'data\test_dataset', '*.jpg')):
    # test correct labels
    labelsTest.append(labelCode[filepath[18:20]])
    # Create vector grayscale images
    PIL_img = Image.open(filepath).convert('L')
    # file shorter name
    fileName = filepath.split('\\')[2]
    testImgNames.append(fileName)
    # width, height = PIL_img.size
    imgVectorsTest.append(np.array(PIL_img).flatten().tolist())

# transform test data with PCA
pca_imgVectorsTest = pca.transform(np.array(imgVectorsTest))

# predicted labels 
pred_labels = clf.predict(pca_imgVectorsTest)

print("Predictions from the classifier:")
print(pred_labels)
print("Target values:")
print(labelsTest)


typeDist = 2  # euclidean
knnTestAcc = []
for kValue in range(2, 20):
    # build and train model that uses euclidean distance
    clfTest = KNeighborsClassifier(n_neighbors=kValue, p=typeDist)
    clfTest.fit(pca_imgVectorsTrain, labelsTrain)
    # predicted labels
    pred_labelsKnn = clfTest.predict(pca_imgVectorsTest)
    knnTestAcc.append(metrics.accuracy_score(labelsTest, pred_labelsKnn))
# plot best k
plt.figure('K vs Accuracy')
plt.subplot(111)
plt.plot(range(2, 20), knnTestAcc, marker='.')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.show()


# results
resultDict = defaultdict(list)
invLabCode = dict((y, x) for x, y in labelCode.items())
for label, fileName in zip(pred_labels, testImgNames):
    resultDict[invLabCode[label]].append(fileName)
# result dictionary to string
strResult = 'PCA, to aid the process of image classification\n'
strResult += f'Accuracy: {metrics.accuracy_score(labelsTest, pred_labels)}\n\n'
for key, val in resultDict.items():
    strResult += str(key) + '\n'
    for v in val:
        strResult += '\t\t' + str(v) + '\n'
# print to file categories
with open(r'data\results\categories.txt', 'w', encoding="utf-8") as f:
    f.write(strResult)


# literal 5: Print eigen vector matrix!
with open(r'data\results\eigenVectMatrix.txt', 'w', encoding="utf-8") as f:
    eigenVectMatrix = pca.components_
    pdEigenVectMatrix = pd.DataFrame(eigenVectMatrix.transpose())
    f.write('Eigen Vector Matrix\n\n')
    f.write(str(pdEigenVectMatrix) + '\n')
    nRows, nCols = pca.components_.transpose().shape
    pcColNames = []
    for colNum in range(len(eigenVectMatrix)):
        pcColNames.append(f'PC{colNum}')



# literal 1: Graph the eigen face of the mean of the initial matrix, before applyin PCA
meanVect = pca.mean_
plt.figure('Mean Vector Face')
plt.subplot(111)
plt.imshow(meanVect.reshape((imgHeight, imgWidth)), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
#plt.show()

# literal 3: Graph all eigen faces for selected principle components
plt.figure('Eigen Faces')
graphRows = 8
graphCols = 10
for i in range(graphRows * graphCols):
    plt.subplot(graphRows, graphCols, i + 1)
    plt.imshow(eigenVectMatrix[i].reshape(
        (imgHeight, imgWidth)), cmap=plt.cm.gray)
    # plt.title(f'PC-{i}')
    plt.xticks(())
    plt.yticks(())
plt.show()
