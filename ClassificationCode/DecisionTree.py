from sklearn import tree
from sklearn import preprocessing
import numpy as np
import pandas as pd
import graphviz 

def load_data(file):                                                    
    return pd.read_csv(file, sep = ",")

movies=load_data("movie.csv")

X=movies.drop(['Movie','Liked'],axis=1)


Y=movies[['Liked']]

enc = preprocessing.OrdinalEncoder()
enc.fit(X)

X_enc = enc.transform(X)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_enc, Y)

tree.plot_tree(clf)


dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("movies")

dot_data = tree.export_graphviz(clf, out_file=None, 
                    feature_names=['Genre','Length','Director','Famous Actors'],  
                      class_names=['Yes','No'],  
                      filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("movies")