#Packages
import sys
import numpy 
import matplotlib
import pandas
import sklearn

#version check
print('Python:{}'.format(sys.version))
print('Numpy:{}'.format(numpy.__version__))
print('matplotlib:{}'.format(matplotlib.__version__))
print('pandas:{}'.format(pandas.__version__))
print('sklearn:{}'.format(sklearn.__version__))


#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import  scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score


#loading the dataset
names=['id','clump_thickness','univorm_cell_size','uniform_cell_shape','marginal_adhesion','single_epithelial_size','bare_nuclei','bland_chromatin','normal_nuceloli','mitoses','class']
dataset=pd.read_csv("data1.csv",names=names)


#taking care of missing data
dataset.replace('?',-99999,inplace=True)
print(dataset.axes)
dataset.drop(['id'],1,inplace=True)

#shape of dataset
print(dataset.shape)

# do dataset visualization
print(dataset.loc[6])
print(dataset.describe())

#plot histogram for each variable
dataset.hist(figsize=(10,10))
plt.show()

#create scatter plot matrix
scatter_matrix(dataset,figsize=(18,18))
plt.show()

#spliting into X and Y
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,9]

#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train ,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

#specifying testing option
seed=8
scoring='accuracy'

#Define model to train
models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC()))

#evaluate each model in turn
results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=10,random_state=seed)
    cv_results=cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print(msg)
    

#make predictions on validation dataset
for name,model in models:
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    print(name)
    print(accuracy_score(y_test,predictions))
    print(classification_report(y_test,predictions))
    
#now come to prediction at Example using KNN
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#prediction on example
example= np.array([[4,2,1,1,1,2,3,2,5]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)
if(prediction==4):
    print("Malignant")
if(prediction==2):
    print("Benign")




