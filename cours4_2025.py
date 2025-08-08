import streamlit as st
from sklearn import datasets 
import numpy as np 

##Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

##Creation de titre
st.header('''
Explore different Classifier
Which on the best ? 
''')

##Mettre la selection à gauche
dataset_name=st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine Dataset'))
st.write(dataset_name)

##Selectionner le classifier
classifier_name=st.sidebar.selectbox('Select le classifieur', ('KNN', 'SVM', 'Random Forest'))

##Selection et affichage dataset

def get_dataset(dataset_name):
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=='Breast Cancer':
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y

X,y=get_dataset(dataset_name)
st.write('Overview of data')
st.dataframe(X)

st.write('Shape of dataset', X.shape)
st.write('Number of classes', len(np.unique(y)))

##Ajouter les paramètres de notre modele

def add_parameter_ui(classifier_name):
    params=dict()
    if classifier_name=='KNN':
        K=st.sidebar.slider('K', 1,15)
        params['K']=K
    elif classifier_name=='SVM':
        C=st.sidebar.slider('C', 0.01, 5.00 )
        params['C']=C
    else:
        max_depth=st.sidebar.slider('max depth', 2,15)
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['max_depth']=max_depth
        params['n_estimators']=n_estimators
    return params
params=add_parameter_ui(classifier_name)

##Mise en place des modeles

def get_classifier(clf_name, params):
    if clf_name=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name=='SVM':
        clf=SVC(C=params['C'])
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    return clf

clf=get_classifier(classifier_name, params)


##Train et test Split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_pred, y_test)
st.write(f'classifier={classifier_name}')
st.write(f'acc={acc}')

##Matrice de confusion
conf_matrix=confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

##PCA 
pca=PCA(2)
x_projete=pca.fit_transform(X_test)


##Graphique
x1=x_projete[:,0]
x2=x_projete[:,1]

fig, (ax1, ax2)=plt.subplots(1,2)

ax1.scatter(x1,x2, c=y_test, cmap='viridis', alpha=0.8)
ax1.set_title('label')
ax2.scatter(x1,x2, c=y_pred, cmap='viridis', alpha=0.8)
ax2.set_title('Prediction')

st.pyplot(fig)

##Exercice : Dans l'application on voit qu'on a la possibilité
##de chosir les hyperparametres
##Merci d'ajouter un texte qui dit : la meilleur valeur du  parametre est X
## et ça donne une accuracy à Y




