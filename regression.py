import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0

    def fit(self, X, y):
        if self.fit_intercept:
            x_train = np.c_[one, X]
            expression = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y
            self.intercept += expression[0]
            self.coefficient = np.append(self.coefficient, expression[1:])
        else:
            x_train = X
            expression = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y
            self.coefficient = np.append(self.coefficient, expression)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[one, X]
            custom_coeff = np.insert(self.coefficient, 0, self.intercept, axis=0)
            return X @ custom_coeff
        else:
            return X @ self.coefficient

    def r2_score(self, y, yhat):
        numerator = np.sum(np.subtract(y, yhat)**2)
        denominator = np.sum(np.subtract(y, np.mean(y))**2)
        return 1 - numerator/denominator

    def rmse(self, y, yhat):
        return np.sqrt(np.sum(np.subtract(y, yhat)**2)/len(y))



f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]
X = np.vstack((f1, f2, f3)).T

c1 = CustomLinearRegression(fit_intercept=True)
one = np.ones([len(f1)], dtype=int).T
c1.fit(X, y)
yh = c1.predict(X)



model = LinearRegression(fit_intercept=True)
model.fit(X, y)
prediction = model.predict(X)
mse = mean_squared_error(y, prediction, squared=False)
r2 = r2_score(y, prediction)
result = {'Intercept': model.intercept_ - c1.intercept,
          'Coefficient': np.subtract(model.coef_, c1.coefficient),
          'R2': r2 - c1.r2_score(y, yh),
          'RMSE': mse - c1.rmse(y, yh)}
print(result)

