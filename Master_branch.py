import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from pylab import rcParams
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def dataframeFormation():

    df = pd.read_csv('final_merged_dataset_mthly2.csv', encoding='latin-1')

    #Finding the number of occurences of each Query string
    row_counts = {}
    for i in df2['searchTerm']:
        if i in row_counts:
            row_counts[i] += 1
        else:
            row_counts[i] = 1


    #Forming a list with the search queries containing max number of entrees
    highfreq_searchStrings = []
    for j in row_counts:
        if row_counts[j] > 450:
            highfreq_searchStrings.append(j)


def dataCleaning(df):
    ATR_std, CTR_std = df[['ATR', 'CTR']].std()
    ATR_mean, CTR_mean = df[['ATR', 'CTR']].mean()

    # data Cleaning
    df_WO_Outliers = df[((df['ATR'] <= (ATR_mean + 2 * ATR_std)) & (df['ATR'] >= (ATR_mean - 1 * ATR_std)))]
    df_WO_Outliers = df_WO_Outliers[((df['CTR'] <= (CTR_mean + 2 * CTR_std)) & (df['CTR'] >= (CTR_mean - 1 * CTR_std)))]

    # scatter plots of data
    rcParams['figure.figsize'] = 5, 5
    sb.set_style('whitegrid')
    # plt.hist(dfsearch['CTR'])
    # plt.plot()
    sb.pairplot(df_WO_Outliers, vars=['CTR', 'ATR', 'conv'], y_vars=['conv'])
    df_WO_Outliers.plot(kind='scatter', x='ATR', y='conv', c=['darkgray'], s=150)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_WO_Outliers['CTR'], df_WO_Outliers['ATR'], df_WO_Outliers['conv'], c='r', marker='o')
    ax.set_xlabel('CTR')
    ax.set_ylabel('ATR')
    ax.set_zlabel('CONV')
    plt.show()
    return df_WO_Outliers


def linearRegression(df):
    # splitting train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(df[['CTR']], df[['conv']], test_size=0.2)

    # training the model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # printing model params
    print(regressor.coef_)
    print(regressor.intercept_)
    print(regressor.score(X_test, y_test))

    # Plotting
    plt.scatter(df['CTR'], df['conv'])
    plt.plot(df['CTR'], regressor.predict(df[['CTR']]), color='blue', linewidth=3)
    plt.show()


def multipleLinearRegression(df):
    x = df.iloc[:, 6:-1].values
    y = df.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Fitting Multiple Linear Regression to the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the test results
    y_pred = regressor.predict(X_test)

    # printing model params
    print(regressor.coef_)
    print(regressor.intercept_)
    print(regressor.score(X_test, y_test))





def supportVectorRegression(df):
    X = df.iloc[:, 6:-1].values
    y = df.iloc[:, 8].values

    '''
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, y_train)

    # printing model params
    # print(regressor.coef_)
    # print(regressor.intercept_)
    # print(regressor.score(X_test,y_test))

    '''
    # Predicting a new result
    y_pred = regressor.predict(6.5)
    y_pred = sc_y.inverse_transform(y_pred)

    '''

    '''
    # Visualising the SVR results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('SVR')
    plt.xlabel('ATR, CTR')
    plt.ylabel('CONV')
    plt.show()

    '''

    from sklearn.model_selection import cross_val_score
    # clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(regressor, X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


