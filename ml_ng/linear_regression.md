# Linear Regression

Simple form of linear regression:

Prediction: y_hat = a + b*x

x element R^n

theta element R^n-1

theta_revised element R ^ n

Hypothesis: h_theta(x) = theta_0 + (theta_1 * x)

Cost Function: J(theta) = (1 / 2 * m) SUM (h_theta(x(i)) - y(i))^2

FOR THIS TO WORK WE MOST USE THETA_REVISED
Gradient Descent: theta_j := theta_j - alpha * (1/m) * SJM((h_theta(x(i)) - y(i)) * x_j(i))
