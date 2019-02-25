import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as logreg
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import glob
import os


def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
          
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]
            

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

X, y = load_data("Dataset")
F = extract_lbp_features(X)



            
# Apply grid and search for SCV classifier


# Create hyperparameter options
hyperparams = {'C':[0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000],'gamma':[1,0.1,0.001,0.0001],
              'kernel':['linear','rbf', 'poly'], 'degree':[1, 2, 3, 4]}

grid_svc = GridSearchCV(SVC(),hyperparams, cv= 5)
        
grid_svc.fit(F,y)

print('the best parameters for SVC classifier using GridSearchCV are ' + str(grid_svc.best_params_))


searchcv_svc = RandomizedSearchCV(SVC(), hyperparams , n_iter = 20, cv= 5)

searchcv_svc.fit(F,y)

print('the best parameters for SVC classifier using RandomizedSearchCV are ' + str(searchcv_svc.best_params_))

# Apply grid and search for logreg classifier


# Create hyperparameter options
hyperparams_grid = {
    "C": [1e-5, 1e-3, 1e-1, 1],
    "fit_intercept": [True, False],
    "penalty": ["l1", "l2"]
}


grid_logreg = GridSearchCV(logreg(),hyperparams_grid, cv= 5)
        
grid_logreg.fit(F,y)

print('the best parameters for logreg classifier using GridSearchCV are ' + str(grid_logreg.best_params_))

hyperparams_dist = {
    "C": stats.beta(1, 3),
    "fit_intercept": [True, False],
    "penalty": ["l1", "l2"]
}


searchcv_logreg = RandomizedSearchCV(logreg(), hyperparams_dist , n_iter = 20, cv= 5)

searchcv_logreg.fit(F,y)

print('the best parameters for logreg classifier using RandomizedSearchCV are ' + str(searchcv_logreg.best_params_))

