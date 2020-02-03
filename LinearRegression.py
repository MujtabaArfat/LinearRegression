#####################################################################################################################
#   CS 6375 - Assignment 1, Linear Regression using Gradient Descent
#   This is a simple starter code in Python 3.6 for linear regression using the notation shown in class.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a file. Of course, you will need to include the file to make sure the code runs.
#         - you can assume the last column will the label column
#   test - test dataset - can be a link to a URL or a file. Of course, you will need to include the file to make sure the code runs.
#         - you can assume the last column will the label column
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer,StandardScaler

class LinearRegression:
    def __init__(self, train):
        np.random.seed(1)
    
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        df=pd.read_csv('datasets.csv',header=0)
        df.drop(labels=['No','X1 transaction date' ],axis=1,inplace=True)
        df.drop_duplicates(keep='first',inplace=True)
        df.dropna(inplace=True)
        #print(df.head(25))
        df.insert(0,'X0',1)
        
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        self.X =  df.iloc[:, 0:(self.ncols -1)].values.reshape(self.nrows, self.ncols-1)
        self.y = df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)
       
        

    


    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function
    def preProcess(self):
        self.X = self.X
        scaler = StandardScaler()
        
        self.X=scaler.fit_transform(self.X)
        self.y=scaler.fit_transform(self.y)
        #Filling of missing values in dataset
        #imputer=Imputer(missing_values='NaN',strategy="mean",axis=0)
        #imputer=imputer.fit(self.X[:,:])
        #MinMax scaling of input features and output vector
        #self.X[:,:]=imputer.transform(self.X[:,:])
        
        
        #scaler=MinMaxScaler(feature_range=(100,200))
        #self.X[:,:]=scaler.fit_transform(self.X)
        #self.y[:,:]=scaler.fit_transform(self.y)
        #print(self.X)
        
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=0)
        pd.DataFrame(self.X_train).to_csv('X_train.csv',index=False)
        
        pd.DataFrame(self.y_train).to_csv('Y_train.csv',index=False)
        pd.DataFrame(self.X_test).to_csv('X_test.csv',index=False)
        pd.DataFrame(self.y_test).to_csv('Y_test.csv',index=False)
        
        
        #print(self.X_train[:5])
        #print("BLAH")
       # print(self.Y_train)
        
    # Below is the training function
    def train(self, epochs = 10, learning_rate = 0.05):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X_train, self.W)
            
            
            # Find error
            error = h - self.y_train
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X_train.T, error)

        return self.W, error

    # predict on test dataset
    def predict(self, test):
        testXDF = pd.read_csv('X_train.csv')
        testYDF = pd.read_csv('Y_train.csv')
        testXDF.insert(0, "X0", 1)
        nrowsx, ncolsx = testXDF.shape[0], testXDF.shape[1]
        nrowsy, ncolsy = testXDF.shape[0], testYDF.shape[1]
        testX = testXDF.iloc[:, 1:].values.reshape(nrowsx, ncolsx-1)
    
        
        
        testY = testYDF.iloc[:, :].values.reshape(nrowsy, 1)
        
        pred = np.dot(testX, self.W)
        error = pred - testY
        mse = (1/(2*nrowsx)) * np.dot(error.T, error)
        return mse


if __name__ == "__main__":
    model = LinearRegression("datasets.csv")
    model.preProcess()
    mse_err=[]
    epoch=10
    learning_rate=0.05
    for i in range(40):
        W, e = model.train(epoch,learning_rate)
        mse = model.predict("test.csv")
        mse_err.append([epoch,learning_rate,mse])
        epoch+=5
        learning_rate+=0.01
        
    print(mse_err)



