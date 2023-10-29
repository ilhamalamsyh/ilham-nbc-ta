import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        X = X.toarray()
        y = y.to_numpy()
        # get number of samples (rows) and features (columns):
        self.n_samples, self.n_features = X.shape
        # get number of uniques classes
        self.n_classes = len(np.unique(y))

        # create three zero-matrices to store summary stats & prior
        self.mean = np.zeros((self.n_classes, self.n_features))
        self.variance = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            # create a subset of data for thr specific class 'c'
            X_c = X[y == c]

            # calculate statistics and update zero matrices, rows=classes, cols=features
            self.mean[c, :] = np.mean(X_c, axis=0)
            self.variance[c, :] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / self.n_samples
    
    def gaussian_density(self, x, mean, var):
        # print('x: ', x)
        # print('mean: ', mean)
        # print('var: ', var)
        print('x_gd: ', x)
        print('mean_type_gd: ', type(mean))
        print('var_gd: ', var)
        # implementation of gaussian density function
        const = 1 / np.sqrt(var * 2 * np.pi)
        print('const_type_gd: ', const)
        proba = np.exp(-0.5 * ((x - mean) ** 2 / var))
        print('const_proba_gd: ', type(proba))
        # print('const_tf: ', const)
        # print('proba: ', proba)
        # print('gd_return: ', const * proba)
        print('const_x_proba: ', const * proba)
        return const * proba
    
    def get_class_probability(self, x):
        # store new posteriors for each class in a single list
        posteriors = list()

        for c in range(self.n_classes):
            # get summary stats and prior
            mean = self.mean[c]
            variance = self.variance[c]
            prior = np.log(self.priors[c])
            # calculate new posterioirs & append to list
            posterior = np.sum(np.log(self.gaussian_density(x, mean, variance)))
            # print('posterior_nih: ', posterior)
            posterior  = prior + posterior
            posteriors.append(posterior)
            
        
        # return the index with the highest class probability
        return np.argmax(posteriors)

    def predict(self, X):
        X = X.toarray()
        # for each sample x in the dataset X
        y_hat = [self.get_class_probability(x) for x in X]
        # print("y_hat: ", y_hat)
        return np.array(y_hat)


    # def fit(self, X, y):
    #     n_samples, n_features = X.shape
    #     self._classes = np.unique(y)
    #     n_classes = len(self._classes)

    #     # calculate mean, variant, and prior for each class
    #     self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
    #     self._var = np.zeros((n_classes, n_features), dtype=np.float64)
    #     self._priors = np.zeros(n_classes, dtype=np.float64)

    #     for idx, c in enumerate(self._classes):
    #         X_c = X[y == c]
    #         self._mean[idx, :] = X_c.mean(axis=0)
    #         self._var[idx, :] = X_c.toarray().var(axis=0)
    #         self._priors[idx] = X_c.shape[0] / float(n_samples)
            
    # def predict(self, X):
    #     y_pred  = [self._predict(x) for x in X.toarray()]
    #     print('y_pred: ', np.array(y_pred))
    #     return np.array(y_pred)
    
    # def _predict(self, x):
    #     posteriors = []

    #     # Calculate posterior probability for each class
    #     for idx, c in enumerate(self._classes):
    #         prior = np.log(self._priors[idx])
    #         print('prior: ', prior)
    #         posterior = np.sum(np.log(self._pdf(idx, x)))
    #         print('posterior: ', posterior)
    #         posterior = posterior + prior
    #         print('posteriorSum: ', posterior)
    #         posteriors.append(posterior)

    #     # return class with the highest posterior
    #     return self._classes[np.argmax(posteriors)]

    # def _pdf(self, class_idx, x):
    #     mean = self._mean[class_idx]
    #     var = self._var[class_idx]
    #     print('mean: ', type(mean))
    #     print('var: ', type(var))
    #     numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    #     denominator = np.sqrt(2 * np.pi * var)
    #     return numerator / denominator

