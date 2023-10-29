import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayesText:
    def fit(self, X, y):
        X = X.tolist()
        y = y.to_numpy()
        print('X_type: ', type(X))
        print('y_type: ', type(y))

        n_samples = len(X)
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Menghitung probabilitas prior untuk setiap kelas
        self.priors = np.zeros(n_classes)
        for idx, c in enumerate(self.classes):
            self.priors[idx] = np.sum(y == c) / float(n_samples)

        # Menghitung jumlah kemunculan setiap kata dalam setiap kelas
        self.word_counts = {}
        self.feature_names = []
        vectorizer = CountVectorizer()
        X_transformed = vectorizer.fit_transform(X)
        self.feature_names = vectorizer.get_feature_names_out()
        for idx, c in enumerate(self.classes):
            X_c = X_transformed[y == c]
            self.word_counts[c] = np.array(X_c.sum(axis=0)).flatten()

    def predict(self, X):
        X = X.tolist()

        y_pred = []
        vectorizer = CountVectorizer(vocabulary=self.feature_names)
        X_transformed = vectorizer.fit_transform(X)
        for i in range(len(X)):
            text = X_transformed[i]
            posteriors = []
            for idx, c in enumerate(self.classes):
                # Menghitung log likelihood menggunakan model Multinomial Naive Bayes
                likelihood = np.sum(np.log((self.word_counts[c] + 1) / (np.sum(self.word_counts[c]) + len(self.feature_names))))
                # Menghitung log posterior dengan menambahkan log likelihood dan log prior
                posterior = np.log(self.priors[idx]) + likelihood
                posteriors.append(posterior)
            # Memilih kelas dengan log posterior tertinggi sebagai prediksi
            y_pred.append(self.classes[np.argmax(posteriors)])
        return y_pred
