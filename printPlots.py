import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_graphviz,export_text
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import seaborn as sns 
import graphviz
## Analisis exploratorio
# Para variables cualitativas
def print_status(var, dataframe):
    f = plt.figure(figsize = (8,4))
    sns.countplot(var, hue="status", data=dataframe)
    plt.show()
    
def print_salary(var, dataframe):
    f = plt.figure(figsize = (8,4))
    for i in np.unique(dataframe[var]):
        sns.kdeplot(dataframe.salary[dataframe[var]==i])
    plt.legend(np.unique(dataframe[var]))
    plt.show() 
    
def print_salary_boxplot(var, dataframe): 
    f = plt.figure(figsize = (8,4))
    sns.boxplot("salary", var, data=dataframe)
    plt.show()
    
# Para variables cuantitativas
def print_status_kernel(var, data):  
    sns.kdeplot(data[var][ data.status=="Placed"])
    sns.kdeplot(data[var][ data.status=="Not Placed"])
    plt.legend(["Placed", "Not Placed"])
    plt.show()
                                        ##Añade la cualitativa*
def print_salary_scatter(var1, dataframe,var2=None):
    f = plt.figure(figsize = (8,4))
    sns.relplot(x=var1, y="salary", hue=var2, style=var2,
                data=dataframe)
    plt.show()
## Modelos 

def print_pretty_confusionMatrix(clf, features, X_test, y_test):
    class_names = features
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                    display_labels=class_names,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)
    plt.show()

# Decision Tree
def print_decisionTree(clf):
    plt.figure(figsize=(12,12)) 
    plot_tree(clf, fontsize=12)
    plt.show()
    
def print_decisionTree_colour(clf,features,classes):
    dot_data = export_graphviz(clf, out_file=None, 
                        feature_names=features,  
                        class_names=classes,  
                        filled=True, rounded=True,  
                        special_characters=True)  
    graph = graphviz.Source(dot_data) 
    return graph  
def print_decisionTree_text(clf,features):
    r = export_text(clf, feature_names=features)
    print(r)
# PCA
def print_varExplicada(modelo_pca, X):
    print('----------------------------------------------------')
    print('Porcentaje de varianza explicada por cada componente')
    print('----------------------------------------------------')
    print(modelo_pca.explained_variance_ratio_)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.bar(
        x      = np.arange(modelo_pca.n_components_) + 1,
        height = modelo_pca.explained_variance_ratio_
    )
    for x, y in zip(np.arange(len(X.columns)) + 1, modelo_pca.explained_variance_ratio_):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )

    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Porcentaje de varianza explicada por cada componente')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza explicada')
    plt.show();

def print_varExpCum(modelo_pca, X):
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    print('------------------------------------------')
    print('Porcentaje de varianza explicada acumulada')
    print('------------------------------------------')
    print(prop_varianza_acum)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(
        np.arange(len(X.columns)) + 1,
        prop_varianza_acum,
        marker = 'o'
    )
    for x, y in zip(np.arange(len(X.columns)) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
        
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_title('Porcentaje de varianza explicada acumulada')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza acumulada')
    plt.show();