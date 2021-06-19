class Models:

    def classify(data,features,labels,smote = True,sparse = True,test_size = 0.2,random_state =42, tune = 'n',cv_folds = 5):
        """
        Here Smote is deault True but if you dont want to apply smote on dataset then turn it off smote = False
        This Function takes data, hues , features, labels as input and performs everything right from Visualisation to Prediction using 11 Models.
        Features:
        Preprocessing
                1) SMOTE
                2) Label Encoding and One-Hot Encoding of categorical data.
                3) Splits data into Test and Train Set
                4) Scaling Data
            Prediction
                1) Logistic Regression
                2) Support Vector Classification
                3) K-Nearest Neighbors
                4) Decision Tree Classifiers
                5) GaussianNB
                6) Stochastic Gradient Descent
                7) Random Forest Classifier
                8) AdaBoost Classifier
                9) Gradient Boosting
                10) Light Gradient Boosting
                11) PassivaAggressive Classifier
            CrossValidation
                1) K-Fold CV
                2) GridSearch CV
            Parameters:
                data : pd.DataFrame
                    Dataset
                hues : str
                    Hues for visualisation
                features : pd.DataFrame
                    Features for Prediction
                labels : pd.DataFrame
                    Labels for prediction
                test_size : int or float
                    Percentage for test set split. Default = 0.2
                random_state : int
                tune : str
                    Whether to enable hypertuning or not. Default = 'n
                cv_folds : int
                    No. of CV Folds. Default 5
        """

        from sklearn.model_selection import GridSearchCV,StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.linear_model import PassiveAggressiveClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import accuracy_score
        from sklearn.naive_bayes import GaussianNB
        from imblearn.over_sampling import SMOTE
        from lightgbm import LGBMClassifier
        from collections import Counter
        from sklearn.svm import SVC
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import warnings
        import scipy

        warnings.simplefilter(action='ignore', category=Warning)

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
        if smote == True:
            print('Applying SMOTE [*]\n')
            sm=SMOTE(k_neighbors=4)
            features,labels=sm.fit_resample(features,labels)
            print('SMOTE Done [',u'\u2713',']\n')

        ## Sparse Matrix ---------------------------------------------------------------------
        if sparse==True:
            if scipy.sparse.issparse(features[()]):
                print('Converting Sparse Features to array []\n')
                features = features[()].toarray()
                print(
                            'Conversion of Sparse Features to array Done [', u'\u2713', ']\n')

            elif scipy.sparse.issparse(labels[()]):
                print('Converting Sparse Labels to array []\n')
                labels = labels[()].toarray()
                print(
                            'Conversion of Sparse Labels to array Done [', u'\u2713', ']\n')

            else:
                print("No, sparce matrix found....")

        ## Splitting ---------------------------------------------------------------------

        print('Splitting Data into Train and Validation Sets [*]\n')

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_size, random_state = random_state)

        print('Splitting Done [',u'\u2713',']\n')

        ## Scaling ---------------------------------------------------------------------
        print('Scaling Training and Test Sets [*]\n')

        sc = StandardScaler()
        X_train = sc.fit_transform(x_train)
        X_val = sc.transform(x_test)
        print('Scaling Done [',u'\u2713',']\n')

        print('Training All Basic Classifiers on Training Set [*] \n')

        parameters_svm= [
        {'kernel': ['rbf'], 'gamma': [0.1, 0.5, 0.9, 1],
            'C': np.logspace(-4, 4, 5)},
        ]
        parameters_lin = [{
        'penalty': ['l1', 'l2', ],
        'solver': ['newton-cg', 'liblinear', ],
        'C': np.logspace(-4, 4, 5),
        }]
        parameters_knn = [{
        'n_neighbors': list(range(0, 11)),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'kd_tree', 'brute'],
        }]
        parameters_dt = [{
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [4,  6,  8,  10,  12,  20,  40, 70],

        }]
        parameters_rfc = [{
        'criterion': ['gini', 'entropy'],
        'n_estimators': [100, 300, 500, 750, 1000],
        'max_features': [2, 3],
        }]
        parameters_xgb = [{
        'max_depth': [4,  6,  8,  10],
        'learning_rate': [0.3, 0.1],
        }]
        parameters_lgbm =  {
        'learning_rate': [0.005, 0.01],
        'n_estimators': [8,16,24],
        'boosting_type' : ['gbdt', 'dart'],
        'objective' : ['binary'],
        }
        paramters_pac = {
            'C': np.logspace(-4, 4, 20)},


        param_nb={}
        parameters_ada={
                'learning_rate': [0.005, 0.01],
                'n_estimators': [8,16,24],
        }
        paramters_sgdc = [{
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['hinge', 'log'],
        'alpha':np.logspace(-4, 4, 20),
        }]

        models =[("LR", LogisticRegression(), parameters_lin),("SVC", SVC(),parameters_svm),('KNN',KNeighborsClassifier(),parameters_knn),
        ("DTC", DecisionTreeClassifier(),parameters_dt),("GNB", GaussianNB(), param_nb),("SGDC", SGDClassifier(), paramters_sgdc),('RF',RandomForestClassifier(),parameters_rfc),
        ('ADA',AdaBoostClassifier(),parameters_ada),('XGB',GradientBoostingClassifier(),parameters_xgb),('LGBN', LGBMClassifier(),parameters_lgbm),
        ('PAC',PassiveAggressiveClassifier(),paramters_pac)]

        results = []
        names = []
        finalResults = []
        accres = []

        for name,model, param in models:
            print('Training {} and Showing Predictions ['.format(str(model)),u'\u2713','] \n')
            model.fit(x_train, y_train)

            model_results = model.predict(x_test)
            accuracy = accuracy_score(y_test, model_results)
            print('Validation Accuracy is :',accuracy)

            print('Applying K-Fold Cross validation on Model {}[*]'.format(name))
            accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=cv_folds, scoring='accuracy')
            print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
            acc = accuracies.mean()*100
            print("Standard Deviation: {:.2f} %\n".format(accuracies.std()*100))
            results.append(acc)
            names.append(name)
            if tune == 'y' and not name == 'GNB':
                print('Grid Search Cross validation for model {} []\n'.format(name))
                cv_params = param
                grid_search = GridSearchCV(
                estimator=model,
                param_grid=cv_params,
                scoring='accuracy',
                cv=cv_folds,
                n_jobs=-1,
                verbose=4,)
                grid_search.fit(X_train, y_train)
                best_accuracy = grid_search.best_score_
                best_parameters = grid_search.best_params_
                print("Best Accuracy for model {}: {:.2f} %".format(name,best_accuracy*100))
                print("Best Parameters: for model {}".format(name), best_parameters)
                print('Grid Search Cross validation Done[',u'\u2713',']\n')
                print('Training {} Done ['.format(str(model)),u'\u2713','] \n')
                print('########################################################################\n')
                accres.append((name,acc, best_parameters))
            elif not tune=='y':
                print('Training {} Done ['.format(str(model)),u'\u2713','] \n')
                print('########################################################################')
                accres.append((name,acc))
        accres.sort(key=lambda k:k[1],reverse=True)
        print("\n The Accuracy of the Models Are:\n ")
        if tune=='y':
            tab = pd.DataFrame(accres, columns= ['Model','Accuracy', 'Best Params'])
        elif not tune=='y':
            tab = pd.DataFrame(accres, columns= ['Model','Accuracy'])
        print(tab)
        sns.barplot(x=tab['Model'], y=tab['Accuracy'], palette='mako');
        print("\n\nModel With Highest Accuracy is: ",accres[0][0],' with an Accuracy of ',tab.iloc[0,1],'% \n\n')
