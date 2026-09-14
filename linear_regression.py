import numpy as np

np.random.seed(42)

X = np.random.rand(100, 1) * 10

true_w, true_b = 3, 5

true_y = true_w * X + true_b + np.random.randn(100, 1) * 2

def predict(X, w, b):
    return X * w + b

w_guess = 0.0
b_guess = 0.0

learning_rate = 0.01
epoch = 1000

def compute_gradients(X, y_true, y_pred):
    n = len(X)
    dw = (2/n) * np.sum(X * (y_pred - y_true))
    db = (2/n) * np.sum(y_pred - y_true)
    return dw, db

for i in range(epoch):
    y_pred = predict(X, w_guess, b_guess)
    dw, db = compute_gradients(X, true_y, y_pred)

    w_guess = w_guess - learning_rate * dw
    b_guess = b_guess - learning_rate * db



print(w_guess, b_guess)


