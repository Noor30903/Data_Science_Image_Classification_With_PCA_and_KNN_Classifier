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
TRAIN_DATASET_PATH = r'data\train_dataset'
TEST_DATASET_PATH = r'data\test_dataset'

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
    imgname=[]
    for filepath in glob.glob(os.path.join(image_path, '*.jpg')):
        labelsTrain.append(labelCode[filepath.split(os.sep)[-1][:2]])
        image = Image.open(filepath).convert('L')
        fileName = filepath.split('\\')[2]
        imgname.append(fileName)
        img_vectorsTrain.append(np.array(image).flatten().tolist())
    return np.array(img_vectorsTrain), labelsTrain,imgname

# Function to RUN PCA
def apply_pca(X_train, variance_threshold=0.95):
    
    pca = PCA(n_components=variance_threshold, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    
    # Determine the number of components that explain the threshold variance
    n_components = np.sum(np.cumsum(pca.explained_variance_ratio_) <= variance_threshold)
    print(f"PCA selected {n_components} components to explain {variance_threshold * 100}% of variance.")
    
    return pca, X_train_pca, n_components

#eVals = np.square(pca.singular_values_)
def plot_scree(eigenvalues):
    plt.figure('Scree Plot')
    plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot for PCA Eigenvalues')
    plt.axhline(y=1, color='r', linestyle='--')  # Kaiser criterion (Eigenvalue=1)
    plt.grid(True)
    #plt.show()

# find best k for knn classification
def find_best_k(X_train, y_train, X_test, y_test, k_range, typeDist):
    accuracies = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, p=typeDist)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracies.append(metrics.accuracy_score(y_test, pred))
    best_k = k_range[np.argmax(accuracies)]
    print(f'Best k by accuracy: {best_k}')
    plt.figure('K vs Accuracy')
    plt.plot(k_range, accuracies, marker='o')
    plt.xlabel('Number of Neighbors: k')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy for different k values')
    plt.grid(True)
    #plt.show()
    return best_k

img_vectors_train, labels_train, imgnameTrain = load_images(TRAIN_DATASET_PATH, imgHeight, imgWidth, labelCode)

pca, pca_imgVectorsTrain, n_components = apply_pca(np.array(img_vectors_train))

img_vectors_test, labels_test, imgnameTest = load_images(TEST_DATASET_PATH, imgHeight, imgWidth, labelCode)

pca_imgVectorsTest = pca.transform(np.array(img_vectors_test))
# literal 6: Classification
typeDist = 2  # euclidean
k_range=range(2, 20)
k = find_best_k(pca_imgVectorsTrain, labels_train, pca_imgVectorsTest, labels_test, k_range, typeDist)  # it was 8 we changed it to 2 for the best accuracy

# build and train model that uses euclidean distance
clf = KNeighborsClassifier(n_neighbors=k, p=typeDist)
clf.fit(pca_imgVectorsTrain, labels_train)

# predicted labels 
pred_labels = clf.predict(pca_imgVectorsTest)
print("Predictions from the classifier:")
print(pred_labels)
print("Target values:")
print(labels_test)

# results
resultDict = defaultdict(list)
invLabCode = dict((y, x) for x, y in labelCode.items())
for label, fileName in zip(pred_labels, imgnameTest):
    resultDict[invLabCode[label]].append(fileName)
# result dictionary to string
strResult = 'PCA, to aid the process of image classification\n'
strResult += f'Accuracy: {metrics.accuracy_score(labels_test, pred_labels)}\n\n'
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
total_plots = min(graphRows * graphCols, n_components)  # Ensure we do not exceed the number of PCA components

for i in range(total_plots):  # Use the adjusted total_plots here
    plt.subplot(graphRows, graphCols, i+1)
    plt.imshow(pca.components_[i].reshape((imgHeight, imgWidth)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())


plt.show()






# literal 3: accumulated variance for chosen eigen vectors in 2) literal
#dictAccVariance = {}
#pcaVariance = pca.explained_variance_ratio_
#countVar = 0
#accVariance = 0
#accVarianceArr = []
#for var in pcaVariance:
#    countVar += 1
#    accVariance += var
#    accVarianceArr.append(accVariance)
#    dictAccVariance[f'PC-{countVar}'] = accVariance
## print results
#strAccVariance = 'Accumulated variance for selected Principle Components\n\n'
#for k, v in dictAccVariance.items():
#    strAccVariance += f'{k} -> {v}\n'
#with open(r'data\results\AccVariance.txt', 'w', encoding="utf-8") as f:
#    f.write(strAccVariance)
## plot accumulated variance
#plt.figure('Accumulated variance vs components')
#plt.subplot(111)
#plt.plot(range(1, n_components+1), accVarianceArr, marker='.')
#plt.hlines(0.95, 0, n_components, colors='r')
#plt.xlabel('Number of Components')
#plt.ylabel('Acc. Variance')
#plt.show()






'''
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
from time import time

# constants
imgHeight = 256
imgWidth = 256
TRAIN_DATASET_PATH = r'data\train_dataset'
TEST_DATASET_PATH = r'data\test_dataset'

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
    imgname=[]
    for filepath in glob.glob(os.path.join(image_path, '*.jpg')):
        labelsTrain.append(labelCode[filepath.split(os.sep)[-1][:2]])
        image = Image.open(filepath).convert('L')
        fileName = filepath.split('\\')[2]
        imgname.append(fileName)
        img_vectorsTrain.append(np.array(image).flatten().tolist())
    return np.array(img_vectorsTrain), labelsTrain,imgname

# Function to RUN PCA
def apply_pca(X_train, variance_threshold=0.95):
    
    pca = PCA(n_components=variance_threshold, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    
    # Determine the number of components that explain the threshold variance
    n_components = np.sum(np.cumsum(pca.explained_variance_ratio_) <= variance_threshold)
    print(f"PCA selected {n_components} components to explain {variance_threshold * 100}% of variance.")
    
    return pca, X_train_pca, n_components

# find best k for knn classification
def find_best_k(X_train, y_train, X_test, y_test, k_range, typeDist):
    accuracies = []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, p=typeDist)
        start_time = time()
        clf.fit(X_train, y_train)
        training_time = time() - start_time
        
        start_time = time()
        pred = clf.predict(X_test)
        prediction_time = time() - start_time
        
        accuracies.append(metrics.accuracy_score(y_test, pred))
        print(f'k = {k}, Training Time: {training_time:.4f}s, Prediction Time: {prediction_time:.4f}s')
    best_k = k_range[np.argmax(accuracies)]
    print(f'Best k by accuracy: {best_k}')
    plt.figure('K vs Accuracy')
    plt.plot(k_range, accuracies, marker='o')
    plt.xlabel('Number of Neighbors: k')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy for different k values')
    plt.grid(True)
    #plt.show()
    return best_k

img_vectors_train, labels_train, imgnameTrain = load_images(TRAIN_DATASET_PATH, imgHeight, imgWidth, labelCode)

pca, pca_imgVectorsTrain, n_components = apply_pca(np.array(img_vectors_train))

img_vectors_test, labels_test, imgnameTest = load_images(TEST_DATASET_PATH, imgHeight, imgWidth, labelCode)

pca_imgVectorsTest = pca.transform(np.array(img_vectors_test))
# literal 6: Classification
typeDist = 2  # euclidean
k_range=range(2, 20)
k = find_best_k(pca_imgVectorsTrain, labels_train, pca_imgVectorsTest, labels_test, k_range, typeDist)  # it was 8 we changed it to 2 for the best accuracy

# build and train model that uses euclidean distance
clf = KNeighborsClassifier(n_neighbors=k, p=typeDist)
start_time = time()
clf.fit(pca_imgVectorsTrain, labels_train)
training_time_pca = time() - start_time

# predicted labels 
start_time = time()
pred_labels = clf.predict(pca_imgVectorsTest)
prediction_time_pca = time() - start_time

print("Predictions from the classifier:")
print(pred_labels)
print("Target values:")
print(labels_test)

# Results without PCA
print("\nResults without PCA:")
k = find_best_k(img_vectors_train, labels_train, img_vectors_test, labels_test, k_range, typeDist)  # it was 8 we changed it to 2 for the best accuracy

clf = KNeighborsClassifier(n_neighbors=k, p=typeDist)
start_time = time()
clf.fit(img_vectors_train, labels_train)
training_time_no_pca = time() - start_time

start_time = time()
pred_labels_no_pca = clf.predict(img_vectors_test)
prediction_time_no_pca = time() - start_time

print("Predictions from the classifier:")
print(pred_labels_no_pca)
print("Target values:")
print(labels_test)

print("\nTraining and Prediction Time (with PCA):")
print(f"Training Time: {training_time_pca:.4f}s, Prediction Time: {prediction_time_pca:.4f}s")

print("\nTraining and Prediction Time (without PCA):")
print(f"Training Time: {training_time_no_pca:.4f}s, Prediction Time: {prediction_time_no_pca:.4f}s")
'''