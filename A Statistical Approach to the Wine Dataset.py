# A Statistical Approach on the Wine Dataset
# Author: Erkan Sahin


# We start by importing necessary libraries for our task
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode,iplot
from scipy.stats import multivariate_normal
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

init_notebook_mode(connected=True)# Here we activate plotly on Jupyter Notebook
plotly.offline.init_notebook_mode()# I activated the notebook mode since I tested my oode on a Jupyter Notebook


# This function makes predictions with respect to probability density functions. We piped our data into 3 different
# pdf function. Here, we compare the unnormalized posteriors for each data point to decide which class each example
# belongs to.
# We get each class separately to get their mean and covariance.
def seperate_class_examples(training_features, training_labels, features_selected):
    type1_indices = training_labels[training_labels == 1].index
    type2_indices = training_labels[training_labels == 2].index
    type3_indices = training_labels[training_labels == 3].index
    type1_samples = training_features.loc[type1_indices, features_selected]
    type2_samples = training_features.loc[type2_indices, features_selected]
    type3_samples = training_features.loc[type3_indices, features_selected]
    return [type1_samples, type2_samples, type3_samples]


def predict(pdf1, pdf2, pdf3):
    predictions = np.zeros(shape=pdf1.shape)
    counter = 0
    for p1, p2, p3 in zip(pdf1, pdf2, pdf3):
        maximum = max(p1, p2, p3)
        if maximum == p1:
            predictions[counter] = 1
        elif maximum == p2:
            predictions[counter] = 2
        elif maximum == p3:
            predictions[counter] = 3
        counter += 1
    return predictions


# This function is used to plot each 2D Gaussian with their density.
def draw_density(gaussian):
    x, y = np.mgrid[0:1:.01, 0:1:.01]
    pos = np.dstack((x, y))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, gaussian.pdf(pos))
    if gaussian == gaussian1:
        plt.title('Distribution of Class 1')
    elif gaussian == gaussian2:
        plt.title('Distribution of Class 2')
    elif gaussian3 == gaussian:
        plt.title('Distribution of Class 3')

    plt.show()


# This function returns mean and covariance of given data.
def gaussian_mle(data):
    mu = data.mean(axis=0)
    var = np.dot((data - mu).T, (data - mu)) / data.shape[0]  # this is slightly suboptimal, but instructive

    return mu, var


# Firstly, we start by reading the data. It is put in a pandas dataframe and columns are renamed for simplicity.
df = pd.read_csv('wine.data.txt', sep=",", header=None)
df.rename(columns={0: 'Type', 1: 'Alcohol', 2: 'Malic acid', 3: 'Ash', 4: 'Alcalinity of ash', 5: 'Magnesium',
                   6: 'Total phenols', 7: 'Flavanoids', 8: 'Nonflavanoid phenols', 9: 'Proanthocyanins',
                   10: 'Color intensity', 11: 'Hue', 12: 'OD280/OD315 of diluted wines', 13: 'Proline'}, inplace=True)
labels = df['Type']
features = df.drop(labels=['Type'], axis=1)

# Here, we apply normalization to our features. Normalization is a technique often applied as part of data preparation
# for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to a common
# scale, without distorting differences in the ranges of values.It is required only when features have different ranges.
# Every dataset does not require normalization. It is required only when features have different ranges. It seems that
# the range for our features varies significantly; so, we should apply MinMaxScaling to our features. We use it to keep
# our values between 0-1.
scaler = MinMaxScaler()
normalized = scaler.fit_transform(features.astype(float))
features = pd.DataFrame(normalized)
features.rename(
    columns={0: 'Alcohol', 1: 'Malic acid', 2: 'Ash', 3: 'Alcalinity of ash', 4: 'Magnesium', 5: 'Total phenols',
             6: 'Flavanoids', 7: 'Nonflavanoid phenols', 8: 'Proanthocyanins', 9: 'Color intensity', 10: 'Hue',
             11: 'OD280/OD315 of diluted wines', 12: 'Proline'}, inplace=True)

# Feature selection is the process of selecting a subset of relevant features for use in model construction It is time
# to do feature selection. Filter feature selection methods apply a statistical measure to assign a scoring to each
# feature. The features are ranked by the score and either selected to be kept or removed from the dataset. The methods
# are often univariate and consider the feature independently, or with regard to the dependent variable. I selected
# features according to the k highest scores from sklearn using chi2 which is also a filter method.
best_features = SelectKBest(score_func=chi2, k=13).fit(features, labels)
df_scores = pd.DataFrame(best_features.scores_)
df_columns = pd.DataFrame(features.columns)

# Let's visualize our feature selection results and see the importance of each feature visually.
f = df_columns.values.tolist()
s = df_scores.values.tolist()
feature_list = []
scores_list = []
for i in range(len(f)):
    feature_list.append(f[i][0])
    scores_list.append(s[i][0])

trace = go.Bar(x=feature_list,
               y=scores_list)
data = [trace]
layout = {'title': 'Feature Importance with respect to chi2',
          'xaxis': {'title': 'Features'},
          'yaxis': {'title': 'Scores'}}
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# We decided on which features to use. I believe Proline and Flavanoids will be a good pair of features.
# Also, we could apply PCA dimensionality reduction algorithm to our features and get two best principal components.
# This approach gives better results but it is not our case in this task.
selected = ['Proline', 'Flavanoids']
selected_features = features[selected]

# Let's split our data to train and set tests. It is very important that dataset is shuffled well to avoid any element
# of bias/patterns in the split datasets before training the ML model. Our data came ordered so we should shuffle our
# dataset during partitioning. Shuffling should not be done in all kinds of data. However, it is better to shuffle our
# dataset.
X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42,

                                                    shuffle=True)

# Now, we will learn mean and covariance of each class in the training set.
type1_samples, type2_samples, type3_samples = seperate_class_examples(X_train, y_train, selected)
mu1, cov1 = gaussian_mle(type1_samples)
mu2, cov2 = gaussian_mle(type2_samples)
mu3, cov3 = gaussian_mle(type3_samples)

# Here we get 3 different gaussian using the mean and covariance values.
gaussian1 = multivariate_normal(mean=mu1, cov=cov1)
gaussian2 = multivariate_normal(mean=mu2, cov=cov2)
gaussian3 = multivariate_normal(mean=mu3, cov=cov3)

# We visualize each gaussian.
draw_density(gaussian1)
draw_density(gaussian2)
draw_density(gaussian3)

# We use each gaussian to calculate unnormalized posteriors.
pdf1 = gaussian1.pdf(X_test)
pdf2 = gaussian2.pdf(X_test)
pdf3 = gaussian3.pdf(X_test)

# It is time to make predictions comparing unnormalized posteriors on each gaussian.
predictions = predict(pdf1, pdf2, pdf3)

# Let's see the performance of our classifier.
accuracy = accuracy_score(predictions, y_test)
print('Accuracy of the classifier is :', "{0:.3f}".format(accuracy))