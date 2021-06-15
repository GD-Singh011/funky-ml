def do_everything(data, hues,features,labels,test_size = 0.2,random_state =42):

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import GaussianNB
    from imblearn.over_sampling import SMOTE
    from lightgbm import LGBMClassifier
    from collections import Counter
    import matplotlib.pyplot as plt
    import pandas_profiling as pp
    from sklearn.svm import SVC
    import plotly.express as px
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import warnings

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
        labels.value_counts().plot.pie();
        plt.show()
        sns.countplot(x=hues, data=data, palette='viridis');
        plt.show()
    
        print("   \n\n\n------------------------- **BOX PLOTS** ------------------------  \n\n\n")
        def boxplot(columns,data):
            fig = px.box(data, x = data[column], color = hues)
            fig.show()
        for column in features:
            boxplot(column,data)
        
        print("   \n\n\n-------------------- **BAR PLOTS** -----------------------------\n\n\n")
        def hist_count(column, data):
            if column in data:
                f, axes = plt.subplots(1,1,figsize=(15,5))
                sns.countplot(x=column, data=data)
                plt.xticks(rotation = 90)
                plt.suptitle(column, fontsize=20)
                plt.show()
            plt.show()
        for column in features.columns:
            hist_count(column,data)
        
        print("   \n\n\n------------------- **DISTRIBUTION PLOTS** ------------------  \n\n\n")

        def boxdistriplot(columnName):
            if not columnName == hues:
                sns.distplot(data[columnName][labels == 1],color="darkturquoise", rug=True)
                sns.distplot(data[columnName][labels == 0], color="lightcoral", rug=True);
                plt.show()
            
        for column in data.columns:
            boxdistriplot(column)
        print("   \n\n\n---------------------------------------- **CORRELATION & PAIRPLOTS**----------------------------------  \n\n\n")     
        def correlation_pairplots(data):
            plt.figure(figsize=(15,15))
            sns.heatmap(data.corr(),annot = True, cmap = 'Blues')
            plt.show()
            print("\n\n\n\n")
                sns.pairplot(data,hue = hues, palette='Greens')
        plt.show()
    
        correlation_pairplots(data)
    visualization(hues,data,features,labels)
   
    #%%    
    print('Checking if labels or features are categorical! [*]\n')
    cat_features=[i for i in features.columns if features.dtypes[i]=='object']
    if len(cat_features) >= 1 :
        index = []
        for i in range(0,len(cat_features)):
            index.append(features.columns.get_loc(cat_features[i]))
        print('Features are Categorical\n')
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), index)], remainder='passthrough')
        print('Encoding Features [*]\n')
        features = np.array(ct.fit_transform(features))
        print('Encoding Features Done [',u'\u2713',']\n')
    if labels.dtype == 'O':
        le = LabelEncoder()
        print('Labels are Categorical [*] \n')
        print('Encoding Labels \n')
        labels = le.fit_transform(labels)
        print('Encoding Labels Done [',u'\u2713',']\n')
    else:
        print('Features and labels are not categorical [',u'\u2713',']\n')
        
    ## SMOTE ---------------------------------------------------------------------
    print('Applying SMOTE [*]\n')
    
    sm=SMOTE(k_neighbors=4)
    features,labels=sm.fit_resample(features,labels)
    print('SMOTE Done [',u'\u2713',']\n')
    
    ## Splitting ---------------------------------------------------------------------
    print('Splitting Data into Train and Validation Sets [*]\n')
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size= test_size, random_state= random_state)
    print('Splitting Done [',u'\u2713',']\n')
    
    ## Scaling ---------------------------------------------------------------------
    print('Scaling Training and Test Sets [*]\n')
    
    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_val = sc.transform(x_test)
    print('Scaling Done [',u'\u2713',']\n')
    
    print('Training All Basic Classifiers on Training Set [*] \n')
    

    models =[("LR", LogisticRegression()),("SVC", SVC()),('KNN',KNeighborsClassifier()),
    ("DTC", DecisionTreeClassifier()),("GNB", GaussianNB()),("SGDC", SGDClassifier()),('RF',RandomForestClassifier()),
    ('ADA',AdaBoostClassifier()),('XGB',GradientBoostingClassifier()),('LGBN', LGBMClassifier()),
    ('PAC',PassiveAggressiveClassifier())]

    results = []
    names = []
    finalResults = []
    accres = []

    for name,model in models:
        model.fit(x_train, y_train)
        model_results = model.predict(x_test)
        acc = accuracy_score(y_test, model_results)
        results.append(acc)
        names.append(name)
        accres.append((name,acc))
    print('Training Compeleted Showing Predictions [',u'\u2713','] \n')
    accres.sort(key=lambda k:k[1],reverse=True)
    print("\n The Accuracy of the Models Are:\n ")
    tab = pd.DataFrame(accres)
    print(tab)
    sns.barplot(x=tab[1], y=tab[0], palette='mako');
    print("\n\nModel With Highest Accuracy is: \n",accres[0],'\n\n')
