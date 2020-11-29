from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class AdverseMediaClassifierLogisticRegression():
    def __init__(self):
        self.cv = TfidfVectorizer(strip_accents='ascii', lowercase=True, stop_words='english')
        self.lr = LogisticRegression(verbose=1, solver='liblinear', random_state=0, C=6, penalty='l2', max_iter=1000)
        
    def fit(self, X_train, y_train):
        """Fit data
        
        Parameters
        ----------
            X_train: array of strings
            y_train: array of labels (0 or 1)
        """
        self.cv = TfidfVectorizer(strip_accents='ascii', lowercase=True, stop_words='english')
        X_train_cv = self.cv.fit_transform(X_train)
        self.lr = self.lr.fit(X_train_cv, y_train)
        
    def predict(self, X_test):
        """Predict class
        
        Parameters
        ----------
        X_test: pandas DataFrame containing 'title' and 'article' column

        Returns
        -------
        array of predicted labels (0 or 1)
        """
        X_test_texts = X_test[['title', 'article']].agg(' '.join, axis=1)
        return self.predict_array(X_test_texts)

    def predict_array(self, X_test_texts):
        """Predict class
        
        Parameters
        ----------
        X_test_texts: array of strings

        Returns
        -------
        array of predicted labels (0 or 1)
        """
        X_test_cv = self.cv.transform(X_test_texts)
        return self.lr.predict(X_test_cv)