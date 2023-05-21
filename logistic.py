from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    @staticmethod
    def sigmoid(t):
        return 1/(1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        t = coef_[0] + np.dot(row, coef_[1:])
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_tr):
        self.coef_ = [0, 0, 0, 0]
        self.mse_error_first = []
        self.mse_error_last = []
        for k in range(self.n_epoch):
            for i, row in enumerate(np.array(X_train)):
                y_hat = self.predict_proba(row, self.coef_)
                if self.fit_intercept:
                    self.coef_[0] -= self.l_rate * (y_hat - y_tr.iloc[i]) * y_hat * (1 - y_hat)
                for j, num in enumerate(self.coef_[1:], start=1):
                    self.coef_[j] = num - self.l_rate * (y_hat - y_tr.iloc[i]) * y_hat * (1 - y_hat) * row[j - 1]

                if k == 0:
                    first_error = ((y_hat - y_tr.iloc[i]) ** 2) / len(np.array(X_train))
                    self.mse_error_first.append(first_error)
                elif k == (self.n_epoch - 1):
                    last_error = ((y_hat - y_tr.iloc[i]) ** 2) / len(np.array(X_train))
                    self.mse_error_last.append(last_error)
        return self.coef_

    def fit_log_loss(self, X_train, y_tr):
        self.coef_ = [0, 0, 0, 0]
        self.logloss_error_first = []
        self.logloss_error_last = []
        n = len(np.array(X_train))
        for k in range(self.n_epoch):
            for i, row in enumerate(np.array(X_train)):
                y_hat = self.predict_proba(row, self.coef_)
                if self.fit_intercept:
                    self.coef_[0] -= (self.l_rate * (y_hat - y_tr.iloc[i]))/len(np.array(X_train))
                for j, num in enumerate(self.coef_[1:], start=1):
                    self.coef_[j] = num - (self.l_rate * (y_hat - y_tr.iloc[i]) * row[j - 1])/n

                if k == 0:
                    first_error = -(y_tr.iloc[i]*np.log(y_hat) + (1-y_tr.iloc[i]) * np.log(1 - y_hat)) / n
                    self.logloss_error_first.append(first_error)
                elif k == (self.n_epoch - 1):
                    last_error = -(y_tr.iloc[i]*np.log(y_hat) + (1-y_tr.iloc[i]) * np.log(1 - y_hat)) / n
                    self.logloss_error_last.append(last_error)
        return self.coef_

    def predict(self, X_test, cut_off=0.5):
        predicitions = []

        for row in np.array(X_test):
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat >= cut_off:
                predicitions.append(1)
            else:
                predicitions.append(0)
        return np.array(predicitions)


data = load_breast_cancer(as_frame=True)
x_data = data.data
y = data.target
perimeter, concave, radius = 'worst perimeter', 'worst concave points', 'worst radius'
X = x_data[[concave, perimeter, radius]].copy()
X[perimeter] = (X[perimeter] - X[perimeter].mean())/X[perimeter].std()
X[concave] = (X[concave] - X[concave].mean())/X[concave].std()
X[radius] = (X[radius] - X[radius].mean())/X[radius].std()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, train_size=0.8, shuffle=True)

c1 = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
c1.fit_log_loss(X_train, y_train)
score_logloss = accuracy_score(y_test, c1.predict(X_test))
c1.fit_mse(X_train, y_train)
score_mse = accuracy_score(y_test, c1.predict(X_test))
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score_sklearn = accuracy_score(y_test, y_pred)
result = {'mse_accuracy': score_mse,
          'logloss_accuracy': score_logloss,
          'sklearn_accuracy': score_sklearn,
          'mse_error_first': c1.mse_error_first,
          'mse_error_last': c1.mse_error_last,
          'logloss_error_first': c1.logloss_error_first,
          'logloss_error_last': c1.logloss_error_last}
print(result)

print("""
answers to the questions:
1) 0.00003
2) 0.00000
3) 0.00153
4) 0.00576
5) expanded
6) expanded
""")

