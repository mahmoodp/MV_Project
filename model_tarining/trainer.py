import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import cv2

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC as SVC
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

from simplelbp import local_binary_pattern
from load_data import load_data 


#  Extract LBP features from all input samples.
#   - R is radius parameter
#    - P is the number of angles for LBP
def extract_lbp_features(X, P = 8, R = 5):
    
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)



# Extract features from input image to feed to classifier
def extract_features(img):
    lbp1 = local_binary_pattern(img, 8, 5)
    feature = np.histogram(lbp1, bins=range(257))[0]
    feature = np.reshape(feature,[1,256])
    return feature

# Load data
X, y = load_data("Dataset")
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))


# Split sample data into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=.2)

# Define teh name of classifiers
names = ["RandomForest", "AdaBoost", "ExtraTrees", "GradientBoosting", "KNeighbors", "LinDiscrimAnalysis", "SVC", "LogisticRegression"  ]

# Define classifiers and hyperparameters
classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),learning_rate=1, n_estimators=400, algorithm="SAMME"),
    ExtraTreesClassifier(n_estimators=1000, max_features=128,n_jobs=1,random_state=0),
    GradientBoostingClassifier(),
    KNN(3),
    LDA(),
    SVC(C=0.001, degree=1, gamma= 0.1, kernel='poly', verbose=False),
    logreg()
        ]


# Loop over each classifier and fit the model to train subset
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    # save the model to disk
    filename = name + '.sav'
    pickle.dump(clf, open(filename, 'wb'))
    #joblib.dump(clf, filename)
    score = clf.score(X_test, y_test)
    print('The score of '+ name + ' classifier is '+ str(score))




'''   

# load the model from disk
loaded_model = joblib.load('SVC.sav')
#loaded_model = pickle.load(open('RandomForest.sav', 'rb'))

img = cv2.imread('em2.jpg')    
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(loaded_model.predict(extract_features(img)))
'''