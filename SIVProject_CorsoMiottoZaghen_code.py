"""                Signal, Image and Video Project                    """
""" An image-statistics-based approach to detecting recaptured images """
"""     Corso Laura 230485, Miotto Sara 232086, Zaghen Olga 224436    """

import numpy as np
from numpy import asarray
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image 
import cv2
from scipy.stats.stats import pearsonr   
import pandas as pd
from pandas.io.parsers import read_csv 
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import time 

start = time.time() # Variable used for the computation of the execution time of the whole algorithm

# ------------ Extraction of the labels ("Recaptured", "Original Captured") and storing in vector Y ------------

dataframe = pd.read_csv("/home/SIVrecapture/NewProject/SignalProjectCorsoMiottoZaghen/labels_final.txt", delimiter='\t', header=None)
dataframe.columns = ['Name', 'Type']
dataframe.to_csv("/home/SIVrecapture/NewProject/SignalProjectCorsoMiottoZaghen/labels_final.csv")
data = read_csv("/home/SIVrecapture/NewProject/SignalProjectCorsoMiottoZaghen/labels_final.csv")
Y = data['Type'] 
Y = np.where(Y.str.contains('Recaptured'), 1, 0) # Vector of 0/1 values, one for each image, where 0 := Original Captured, 1 := Recaptured -> LABELS


# ------------ Extraction of each image and computation of its feature vector ------------

file_name = data['Name']
path = "/home/SIVrecapture/NewProject/SignalProjectCorsoMiottoZaghen/AllImages/"
X = np.zeros((2, 28)) # Matrix of dimension [n_images, 28], where row i contains the feature vector of the i-th image

for name in file_name:
    original = Image.open(path + name).convert('L')
    ratio = original.width/original.height
    resized = original.resize((2048, math.ceil(2048/ratio))) # Resizing of the image so that the width is equal to 2048 pixels
    mat = asarray(resized)

    # ---------------- Creation of the LPF f1 ----------------
    f1 = np.zeros((3, 3))
    f1[1][0] = 1/2
    f1[1][2] = 1/2

    # ---------------- Creation of the LPF f2 ----------------
    f2 = np.zeros((3, 3))
    f2[0][1] = 1/2
    f2[2][1] = 1/2

    # ---------------- Convolution resized * f1 ----------------
    conv1 = signal.convolve2d(mat, f1, mode = 'same')

    # ---------------- Computation of residue image 1 ----------------
    # Residue = trim(image - filtered_image)
    imdiff1 = mat - conv1
    r1 = np.delete(imdiff1, [0, -1], 0) # Trim function
    r1 = np.delete(r1, [0, -1], 1) # Trim function

    # ---------------- Convolution resized * f2 ----------------
    conv2 = signal.convolve2d(mat, f2, mode = 'same')

    # ---------------- Computation of residue image 2 ----------------
    # Residue = trim(image - filtered_image)
    imdiff2 = mat - conv2
    r2 = np.delete(imdiff2, [0, -1], 0)
    r2 = np.delete(r2, [0, -1], 1)


    # ---------------- Visualization of the original image compared to the two residues thresholded1 and thresholded2 ----------------

    # We define thresholded1 starting from r1 by taking its pixel-wise absolute values and performing thresholding at 15, for better visualization
    r1_visual = np.abs(r1)
    ret, thresholded1 = cv2.threshold(r1_visual, 15, 255, cv2.THRESH_TRUNC)

    # We define thresholded2 starting from r2 by taking its pixel-wise absolute values and performing thresholding at 15, for better visualization
    r2_visual = np.abs(r2)
    ret, thresholded2 = cv2.threshold(r2_visual, 15, 255, cv2.THRESH_TRUNC)

    """
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(original, cmap=plt.cm.gray)
    axes[0].set_title("Original")
    axes[1].imshow(thresholded1, cmap=plt.cm.gray)
    axes[1].set_title("Residue 1")
    axes[2].imshow(thresholded2, cmap=plt.cm.gray)
    axes[2].set_title("Residue 2")
    plt.show()
    """


    # ---------------- Computation of the correlation coefficients on Residue Image 1 ----------------

    # We compute the matrix V1 containing the vector V1(i, j) in position (i, j).
    # Each vector V1(i, j) contains the residue value with the same index (i, j) from each patch, concatenated.
    V1 = [[[] for j in range(5)] for i in range(5)]
    for r in range(0, 5):
        for c in range(0, 5):
            subm = r1[r:(r1.shape[0]-4+r), :]
            subm = subm[:, c:(2046-4+c)]
            V1[r][c] = subm.flatten()

    # We compute the matrix C1 containing the correlation coefficients c(i, j) in position (i, j).
    C1 = [[0.0 for j in range(5)] for i in range(5)]

    # We compute the correlation coefficients c(i, j) through the command pearsonr
    for i in range(0, 5):
        for j in range(0, 5):
            c, not_useful = pearsonr(V1[2][2], V1[i][j])
            C1[i][j] = c


    # ---------------- Computation of the correlation coefficients on Residue Image 2 ----------------

    # We compute the matrix V2 containing the vector V2(i, j) in position (i, j).
    # Each vector V2(i, j) contains the residue value with the same index (i, j) from each patch, concatenated.
    V2 = [[[] for j in range(5)] for i in range(5)]
    for r in range(0, 5):
        for c in range(0, 5):
            subm = r2[r:(r2.shape[0]-4+r), :]
            subm = subm[:, c:(2046-4+c)]
            V2[r][c] = subm.flatten()

    # We compute the matrix C2 containing the correlation coefficients c(i, j) in position (i, j).
    C2 = [[0.0 for j in range(5)] for i in range(5)]

    # We compute the correlation coefficients c(i, j) through the command pearsonr
    for i in range(0, 5):
        for j in range(0, 5):
            c, not_useful = pearsonr(V2[2][2], V2[i][j])
            C2[i][j] = c


    # ---------------- Alternative implementation for the computation of the correlation coefficients ----------------

    # An alternative is computing c(i, j) directly using the formula, but this method is much slower than the previous one
    """
    mean33 = np.mean(V[2][2])
    for i in range(0, 5):
        for j in range(0, 5):
            #print("i : ", i)
            #print("j : ", j)
            l1 = np.array([]) # p_(2, 2)^(k) - mean(p_(2, 2))
            l2 = np.array([]) # p_(i, j)^(k) - mean(p_(i, j))
            for k in range(0, K):
                #print(" k : ", k)
                l1 = np.append(l1, (V[2][2][k] - mean33))
                l2 = np.append(l2, (V[i][j][k] - np.mean(V[i][j])))
            print("end k-th cycle")
            num = np.sum(l1*l2)
            den = np.sqrt((np.sum(l1*l1))*(np.sum(l2*l2)))
            C[i][j] = num / den 
    """

    # ---------------- Computation of the final 28-dimensional feature vector through C1 and C2 ----------------
    feature_v = C1[0] + C1[1][1:] + C1[2][3:] + C1[3][3:] + [C1[4][4]] + C2[0] + C2[1][1:] + C2[2][3:] + C2[3][3:] + [C2[4][4]] 

    X = np.append(X, [feature_v], axis=0)

X = X[2:,:] # The first two rows of X are deleted because they were added just for implementational reasons


# ---------------- Training of the SVM, classification, testing and accuracy computation ----------------

# 50 runs needed for the computation of the classification accuracy mean
overall_acc = 0.0 # Sum of the overall accuracies obtained in each run
original_acc = 0.0 # Sum of the accuracies for original images obtained in each run
recaptured_acc = 0.0 # Sum of the accuracies  for recaptured images obtained in each run
for i in range(50):
    # Subdivision of data between training set and test set.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.13, random_state = i) # random_state initialized differently at each iteration in order to have different random splitting on the whole set
    
    # Set the parameters by cross validation
    tuned_parameters = [
        {"kernel": ["rbf"], "gamma": np.linspace(2**(-15), 2**3, 100), "C": np.linspace(2**(-5), 2**(15), 100)}
    ]
    
    # 5-fold cross validation for the training of the SVM and the estimation of the best values for the hyperparameters C and gamma.
    # The best values for C and gamma are chosen through a grid search in order to maximize the accuracy of the classifier.
    classifier = GridSearchCV(SVC(), tuned_parameters, scoring="accuracy")  
    classifier.fit(X_train, Y_train) # Training of the classifier and fitting of the hyperparameters

    print("Best parameters set found on development set:")
    print(classifier.best_params_)
    print()

    y_true, y_pred = Y_test, classifier.predict(X_test) # Testing of the classifier 
    overall_single_acc = accuracy_score(y_true, y_pred) # Computation of the overall accuracy 
    [original_single_acc, recaptured_single_acc] = recall_score(y_true, y_pred, average = None, labels = [0, 1]) # Computation of the accuracies for original and recaptured images 

    print("Overall accuracy: ", overall_single_acc)
    print("Original captured accuracy: ", original_single_acc)
    print("Recaptured accuracy: ", recaptured_single_acc)

    overall_acc = overall_acc + overall_single_acc
    original_acc = original_acc + original_single_acc
    recaptured_acc = recaptured_acc + recaptured_single_acc


overall_acc_mean = overall_acc/50 # Mean of the overall classification accuracies obtained in the previous loop
original_acc_mean = original_acc/50 # Mean of the originals' classification accuracies obtained in the previous loop
recaptured_acc_mean = recaptured_acc/50 # Mean of the recaptured images' classification accuracies obtained in the previous loop
print("Overall accuracy mean: ", overall_acc_mean)
print("Original captured accuracy mean: ", original_acc_mean)
print("Recaptured accuracy mean: ", recaptured_acc_mean)

end = time.time() # Variable used for the computation of the execution time of the whole algorithm
print("Total execution time (in minutes): ", (end-start)/60)