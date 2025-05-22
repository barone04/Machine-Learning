import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')
df= pd.read_csv("pd_speech_features.csv")
df.head()
df['class'].value_counts()

Y = df["class"]
X = df.drop(["id", "class"], axis=1)

X = StandardScaler().fit_transform(X)
pca = PCA()
result = pca.fit_transform(X)

# Remember what we said about the sign of eigen vectors that might change ?
pc1 = - result[:,0]
pc2 = - result[:,1]

plt.figure(figsize=(10,10))
sns.scatterplot(x=pc1, y=pc2, hue=Y, palette=["blue", "yellow"])
plt.show()

def logistic_pca1(X, Y):
    # Decrease data using PCA
    pca1 = PCA(n_components=200)
    x_pca = pca1.fit_transform(X)
    Xnew = pd.DataFrame(x_pca)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(Xnew, Y, train_size=500, random_state=42)

    # Logistic regression model
    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_train, y_train)

    # Make predictions
    pred = logreg.predict(X_test)

    # Calculate accuracy
    acur = accuracy_score(y_test, pred)
    print("Question2 Accuracy: ", acur)
logistic_pca1(X, Y)

def logistic_pca2(X, Y):
    # split train&test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=500, random_state=42)

    # Decrease data
    pca1 = PCA(n_components=200)
    x_train_pca = pca1.fit_transform(X_train)
    x_test_pca = pca1.transform(X_test)

    # logistic regression
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(x_train_pca, y_train)

    pred = log_reg.predict(x_test_pca)

    # Accurracy3
    print("Question3 Accuracy: ", accuracy_score(y_test, pred))
logistic_pca2(X, Y)

def standardScaler_pca1(X, Y):
    # Standardize data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # Doing PCA
    pca = PCA()
    pca.fit(x_scaled)

    # Calculate the amount of information retained
    # Determine the number of principal components needed to retain at least 80% of the variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.80) + 1

    # Doing PCA with new data
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(x_scaled)
    Xnew = pd.DataFrame(X_reduced)
    print(f"Question4.1 New size of data after dimensionality reduction: {X_reduced.shape[1]}")

    # Split train & test
    X_train, X_test, y_train, y_test = train_test_split(Xnew, Y, train_size=500, random_state=42)

    # Logistic regression model
    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_train, y_train)

    # Make predictions
    pred = logreg.predict(X_test)

    # Calculate accuracy
    acur = accuracy_score(y_test, pred)
    print("Question4.2 Accuracy: ", acur)
standardScaler_pca1(X, Y)

def classifier_pca_comparison(X, Y):
    # Split 4:2 train&test on original data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Naive Bayes model on original data
    gaus = GaussianNB()
    gaus.fit(X_train, y_train)
    y_pred_NB = gaus.predict(X_test)
    acur_NB1 = accuracy_score(y_test, y_pred_NB)

    # Logistic Regression model on original data
    lorg = LogisticRegression(max_iter=500)
    lorg.fit(X_train, y_train)
    y_pred_LG = lorg.predict(X_test)
    acur_LG1 = accuracy_score(y_test, y_pred_LG)

    print("Question5: ")
    print("Accuracy of Naive Bayes (before PCA): ", acur_NB1)
    print("Accuracy of Logistic Regression (before PCA): ", acur_LG1)

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Doing PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Calculate the amount of information retained
    # Determine the number of principal components needed to retain at least 80% of the variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.80) + 1

    # Doing PCA with new data
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    Xnew = pd.DataFrame(X_reduced)

    # split train/test after PCA
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(Xnew, Y, test_size=0.2, random_state=42)

    # Naive Bayes model after PCA
    gaus.fit(X_train_pca, y_train_pca)
    y_pred_NB_pca = gaus.predict(X_test_pca)
    acur_NB2 = accuracy_score(y_test_pca, y_pred_NB_pca)

    # Logistic Regression model after PCA
    lorg.fit(X_train_pca, y_train_pca)
    y_pred_LG_pca = lorg.predict(X_test_pca)
    acur_LG2 = accuracy_score(y_test_pca, y_pred_LG_pca)

    print("\nAccuracy of Naive Bayes (after PCA): ", acur_NB2)
    print("Accuracy of Logistic Regression (after PCA): ", acur_LG2)

    # Calculate the change in accuracy
    change_NB = acur_NB2 - acur_NB1
    change_LG = acur_LG2 - acur_LG1

    print("\nChange in accuracy for Naive Bayes: ", change_NB)
    print("Change in accuracy for Logistic Regression: ", change_LG)

    # Compare changes
    if abs(change_NB) > abs(change_LG):
        print("\nNaive Bayes model has greater accuracy change.")
    elif abs(change_NB) < abs(change_LG):
        print("\nLogistic Regression model has greater accuracy change.")
    else:
        print("\nBoth models have the same accuracy change.")
classifier_pca_comparison(X, Y)





