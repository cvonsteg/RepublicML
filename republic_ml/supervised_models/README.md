# Supervised Learning Models

A class of models which 'learn' from classified, labelled data.  These models fit a function based on input -> output mappings, and this function may then be used to answer questions on future, unseen observations of this data.  

## Linear regression
A simple model which derives the linear equation, to generate a line-of-best-fit, for a dependent variable ($y$) with respect to an independent variable ($x$) for a given data set.  The line can then be used to estimate the value of a point for which no data yet exists (commonly referred to as $\hat{y}$)  

In it's simplest form, linear regression involves calculating the `slope` and `y-intercept` values of the line-of-best-fit, to populate the equation:

$$
\hat{y} = a + bx
$$

Where coefficients $a$ and $b$ are defined as:

$$
a = \frac{(\Sigma y)(\Sigma x^2) - (\Sigma x)(\Sigma xy)}{n(\Sigma x^2) - (\Sigma x)^2}
$$

$$
b = \frac{n(\Sigma xy) - (\Sigma x)(\Sigma y)}{n(\Sigma x^2) - (\Sigma x)^2}
$$
