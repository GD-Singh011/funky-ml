def do_everything(data, hues,features,labels,test_size = 0.2,random_state =42):

    import plotly.express as px
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings
    import pandas_profiling as pp
    from collections import Counter
    warnings.simplefilter(action='ignore', category=Warning)
    
    print("Printing First 5 Rows : \n\n")
    print(data.head())
    print("\n\n Getting the Data Types: \n\n")
    print(data.info())
    print("\n\nChecking the Shape Of Data: \n\n ")
    print(data.shape)
    print("\n\nChecking The Null Values [*] \n\n")
    if (data.isnull().any().sum() != 0):
        print("There are empty values in the data so we will remove them....\n\n")
        print(data.dropna(inplace = True))
    else:
        print("There are No null Values in the Data....\n")

    print("\n\nSummarizing the Data \n\n ")
    print(data.describe())

    # Data Visualization
    def visualization(hues,data,features,labels):
        print("\n---------------DATA VISUALIZATION----------------\n")
        print("\nFor Target Value\n")
        y.value_counts().plot.pie();
        plt.show()
        sns.countplot(x=hues, data=data, palette='viridis');
        plt.show()
    #%%
        print("\nPlotting Box-Distri Graphs\n")
        def boxdistriplot(columnName):
            if not columnName == hues:
                sns.catplot(x=hues, y=columnName, data=data, kind="box");
                plt.show()
                sns.distplot(data[columnName][labels == 1],color="darkturquoise", rug=True)
                sns.distplot(data[columnName][labels == 0], color="lightcoral", rug=True);
                plt.show()
            
        for column in data.columns:
            boxdistriplot(column)
    #%%
        print("Printing Box-hist Graphs\n")
        def boxhistplot(columns,data):
            fig = px.histogram(data, x = data[column], color = hues)
            fig.show()
            fig2 = px.box(data, x = data[column], color = hues)
            fig2.show()
        for column in features:
            boxhistplot(column,data)
    #%%

    visualization(hues,data,features,labels)
   
    #%%    
    global accuracy_scores
    
    ## Time Function ---------------------------------------------------------------------
    import time
    start = time.time()
    print("Started Predictor \n")
    
    ## CHECKUP ---------------------------------------------------------------------
    if not isinstance(features, pd.DataFrame) and not isinstance(labels, pd.Series) :
        print('TypeError: This Function take features as Pandas Dataframe and labels as Pandas Series. Please check your implementation.\n')
        end = time.time()
        print(end - start)
        return
    
    ## Encoding ---------------------------------------------------------------------
    print('Checking if labels or features are categorical! [*]\n')
    cat_features=[i for i in features.columns if features.dtypes[i]=='object']
    if len(cat_features) >= 1 :
        index = []
        for i in range(0,len(cat_features)):
            index.append(features.columns.get_loc(cat_features[i]))
        print('Features are Categorical\n')
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), index)], remainder='passthrough')
        print('Encoding Features [*]\n')
        features = np.array(ct.fit_transform(features))
        print('Encoding Features Done [',u'\u2713',']\n')
    if labels.dtype == 'O':
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        le = LabelEncoder()
        print('Labels are Categorical [*] \n')
        print('Encoding Labels \n')
        labels = le.fit_transform(labels)
        print('Encoding Labels Done [',u'\u2713',']\n')
    else:
        print('Features and labels are not categorical [',u'\u2713',']\n')
        
    
    ## SMOTE ---------------------------------------------------------------------
    print('Applying SMOTE [*]\n')
    from imblearn.over_sampling import SMOTE
    sm=SMOTE(k_neighbors=4)
    features,labels=sm.fit_resample(features,labels)
    print('SMOTE Done [',u'\u2713',']\n')
    
    ## Splitting ---------------------------------------------------------------------
    print('Splitting Data into Train and Validation Sets [*]\n')
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size= test_size, random_state= random_state)
    print('Splitting Done [',u'\u2713',']\n')
    
    ## Scaling ---------------------------------------------------------------------
    print('Scaling Training and Test Sets [*]\n')
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    print('Scaling Done [',u'\u2713',']\n')
    
    ## Modelling -------------------------------------------------------------------------
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import SVC,NuSVC
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.model_selection import GridSearchCV,StratifiedKFold
    print('\n\n\n----------------------------Training All Basic Classifiers on Training Set [*]---------------------------\n\n\n')
    
    models =[("LR", LogisticRegression()),("SVC", SVC()),('KNN',KNeighborsClassifier()),
    ("DTC", DecisionTreeClassifier()),("GNB", GaussianNB()),("SGDC", SGDClassifier()),
    ("Perc", Perceptron()),("NC",NearestCentroid()),("Ridge", RidgeClassifier()),
    ("NuSVC", NuSVC()),("BNB", BernoulliNB()),('RF',RandomForestClassifier()),
    ('ADA',AdaBoostClassifier()),('XGB',GradientBoostingClassifier()),
    ('PAC',PassiveAggressiveClassifier())]

    results = []
    names = []
    finalResults = []
    accres = []

    for name,model in models:
        model.fit(X_train, y_train)
        model_results = model.predict(X_val)
        acc = accuracy_score(y_val, model_results)
        results.append(acc)
        names.append(name)
        accres.append((name,acc))
    print('\n\n\n----------------------------Training Compeleted Showing Predictions.... [*]---------------------------\n\n\n')
    accres.sort(key=lambda k:k[1],reverse=True)
    print("\n The Accuracy of the Models Are: ")
    print(accres)
    print("\n\nModel With Highest Accuracy is: ",accres[0])