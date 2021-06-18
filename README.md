# Funky-ml

Funky-ML takes data, hues , features, labels as input and performs everything right from Visualisation to Prediction using 11 Models.

    Features:
        Visualisations
            1) Bar plots
            2) Box Plots
            3) Distribution Plots
            4) Correlation and Pairplots
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
                Whether to enable hypertuning or not. Default = 'n'
            cv_folds : int
                No. of CV Folds. Default 5
                
         Example:
              from funkyml.Funky import funkify
              dataset = pd.read_csv('XYZ.csv')
              features = dataset.iloc[:, :-1]
              lables = dataset.iloc[:, -1]
              funkify(dataset , 'hue' , features, labels)
              
