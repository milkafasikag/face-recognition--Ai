import os
import matplotlib.cm as cm
import numpy as np
from matplotlib import pylab as plt
from matplotlib import pyplot as plt1
from sklearn.linear_model import LogisticRegression
from PIL import Image
from sklearn.decomposition import PCA  

def get_feature_matrix(path, num, average_face, eigen_face):
    testing_labels, testing_data = [], []

    with open(path) as f:
        for line in f:
            im = Image.open(line.strip().split()[0])
            im = np.array(im)
            testing_data.append(im.reshape(2500,))
            testing_labels.append(line.strip().split()[1])
# Define empty lists for train_data and train_labels
train_data, train_labels = [], []
with open('faces/train.txt') as f:
    for line in f:
        im = Image.open(line.strip().split()[0])
        im = np.array(im)
        train_data.append(im.reshape(2500,))
        train_labels.append(line.strip().split()[1])

# Convert train_data and train_labels to NumPy arrays
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)
plt.imshow(train_data[10, :].reshape(50, 50), cmap=cm.Greys_r)
plt.title('The 10th Data')
average_face = np.mean(train_data, axis=0)
plt.figure()
plt.title('Average Face')
plt.imshow(average_face.reshape(50, 50), cmap=cm.Greys_r)
mean_subtracted_data = train_data - average_face
plt.figure()
plt.title('Mean Subtracted Face')
plt.imshow(mean_subtracted_data[10, :].reshape(50, 50), cmap=cm.Greys_r)
pca = PCA(n_components=10)
eigenfaces = pca.fit_transform(mean_subtracted_data)
plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    eigenface = pca.components_[i, :].reshape(50, 50)
    plt.imshow(eigenface, cmap=cm.Greys_r)
plt.title('Top 10 Eigen Faces')
plt.show()

low_ranx_approximation = np.zeros(mean_subtracted_data.shape)
error = np.zeros(200)
for r in range(1, 200):
    print('Now evaluating r = '+str(r))
    Zeta = np.zeros((r,r))
    for i in range(0, r):
        Zeta[i, i] = Sigma[i]
    low_ranx_approximation = (U[:,:r].dot(Zeta)).dot(Vt[:r,:])
    for i in range(0, mean_subtracted_data.shape[0]):
        for j in range(0, mean_subtracted_data.shape[1]):
            aij = (low_ranx_approximation[i][j] - mean_subtracted_data[i][j])
            error[r-1] = error[r-1] + aij*aij
    error[r-1] = np.sqrt(error[r-1])
plt1.figure()
plt1.plot(np.linspace(1,200,200), error)
plt1.title('Error vs r')
plt1.xlabel('r ->')
plt1.ylabel('||X-X_r|| ->')
plt1.show()


r = 10
print('r = ' + str(r))
F_feature_matrix_train = mean_subtracted_data.dot(pca.components_[:r, :].T)
F_feature_matrix_test, test_labels = get_feature_matrix('faces/test.txt', r, average_face, pca.components_)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(F_feature_matrix_train, train_labels)

test_acc = logistic_regression_model.score(F_feature_matrix_test, test_labels)
print('Testing Accuracy : '+str(test_acc*100)+' %')

Accuracy = np.zeros(200)
for r in range(1,200):
    F_feature_matrix_test, test_labels = get_feature_matrix('faces/test.txt', r, average_face, Vt)
    F_feature_matrix_train = mean_subtracted_data.dot(np.transpose(Vt[:r, :]))
    logistic_regression_model.fit(F_feature_matrix_train, train_labels)
    Accuracy[r-1] = logistic_regression_model.score(F_feature_matrix_test, test_labels)*100

plt1.figure()
plt1.title('Accuracy vs r')
plt1.xlabel('r ->')
plt1.ylabel('Accuracy(%) ->')
plt1.plot(np.linspace(1,200,200),Accuracy)
plt1.show()


def Irecognize_image(image_path, pca, logistic_regression_model):
    im = Image.open(image_path)
    im = np.array(im)
    
    # Perform dimensionality reduction (PCA) on the image
    im_pca = pca.transform(im.reshape(1, -1))
    
    # Use the trained model to predict the label
    prediction = logistic_regression_model.predict(im_pca)
    return prediction[0]

# Provide the path to the image you want to recognize
image_path = 'path_to_your_image.png'

# Call the recognition function with the image path
recognized_label = Irecognize_image(image_path, pca, logistic_regression_model)

# Print the recognized label
print('Recognized Label:', recognized_label)
