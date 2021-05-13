import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

%matplotlib inline
'''
iris=pd.read_csv('../SVM CODE/IRIS.csv')

iris.head()

sns.pairplot(data=iris, hue='class', palette='Set2')

x=iris.iloc[:,:-1]
y=iris.iloc[:,4]
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)

model=SVC()
model.fit(x_train, y_train)

pred=model.predict(x_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))
'''

cancer = load_breast_cancer()

oddf_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target_names']))
                       
                       
df_cancer.head()