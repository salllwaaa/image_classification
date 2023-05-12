from PIL import Image
import numpy as np
from skimage.feature import hog
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from skimage.transform import resize
# Set the directory path
dir_path = 'C:/Users/salwa/OneDrive/Documents/pattern/Lab8/Assignment dataset/train'

# Get a list of all files and folders in the directory
subfolders = os.listdir(dir_path)

x_train=[]
y_train=[]


for folder in subfolders:
    for img in os.listdir(os.path.join(dir_path, folder)):
        if img.endswith(".jpg") or img.endswith(".png"):

            image=Image.open(os.path.join(dir_path,folder,img))
            image = image.resize((128,64))

            img_array = np.array(image)

            if len(img_array.shape) == 2:
                img_array = np.dstack([img_array] * 3)

            #print(img_array.shape)
            fd, hog_image = hog(img_array, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            x_train.append(fd)
            y_train.append(folder)


#print(y_train)
X=np.array(x_train)
Y=np.array(y_train)


C = 0.001 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X, Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)


#for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
 #   predictions = clf.predict(X)
  #  accuracy = np.mean(predictions == Y)
   # print("accuracy",accuracy)


lr_ovo = OneVsOneClassifier(LogisticRegression()).fit(X, Y)
lr_ovr = OneVsRestClassifier(LogisticRegression()).fit(X, Y)


# model accuracy for Logistic Regression model
accuracy = lr_ovr.score(X, Y)
#print('OneVsRest Logistic Regression accuracy: ' + str(accuracy))
accuracy = lr_ovo.score(X, Y)
#print('OneVsOne Logistic Regression accuracy: ' + str(accuracy))





##############testing#######################


# Set the directory path for test images
test_dir_path = 'C:/Users/salwa/OneDrive/Documents/pattern/Lab8/Assignment dataset/test'

# Get a list of all files and folders in the directory
test_subfolders = os.listdir(test_dir_path)

x_test=[]
y_test=[]

for folder in test_subfolders:
    for img in os.listdir(os.path.join(test_dir_path, folder)):
        if img.endswith(".jpg") or img.endswith(".png"):
            image=Image.open(os.path.join(test_dir_path,folder,img))
            image = image.resize((128,64))
            # Convert the image to a NumPy array
            img_array = np.array(image)
            # Expand grayscale images to color images
            if len(img_array.shape) == 2:
                img_array = np.dstack([img_array] * 3)

            fd, hog_image = hog(img_array, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            x_test.append(fd)
            y_test.append(folder)

X_test=np.array(x_test)
Y_test=np.array(y_test)

best_acc=0
model=""
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    if accuracy>best_acc:
        best_acc=accuracy
        model=clf.__class__.__name__
    print("accuracy",accuracy)
print("best accuracy",best_acc,f"for Classifier : {model}")


# Predict using logistic regression models
lr_ovo_pred = lr_ovo.predict(X_test)
print('OneVsRest Logistic Regression accuracy: ' + str(accuracy))
lr_ovr_pred = lr_ovr.predict(X_test)
print('OneVsOne Logistic Regression accuracy: ' + str(accuracy))

