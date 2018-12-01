# Machine Learning from Scratch

_"...if your starting point is unknown, and your end-point and intermediate stages are woven together out of unknown material, there may be coherence, but knowledge is completely out of the question..."_ - Plato (Repbulic)

Whilst newer ML libraries and packages make models out-of-the-box, accessible, and easy to use, they do not necessarily lend themselves to understanding the underlying components.  This project is an (ambitious) attempt and learning-by-doing...from scratch.

I will attempt to build common machine learning models/algorithms from scratch using primarily base Python, and occasionally the likes of Numpy for more efficient linear algebra operations.  The goal is not to build something novel or faster, or more efficient.  Instead I want to do as much of the modelling myself, from first principles, in an attempt to understand the underlying mathematics, assumptions, and python data model at a far deeper level than I do at the offset.   

## Model Classes

Models will be split into 4 primary classes:

1) Supervised Regression

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/438px-Linear_regression.svg.png" width="220" height="150">

i) Simple Linear Regression

General Formula: Y = a + bX

<img src="https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2009/11/linearregressionequations.bmp" width="220" height="135">

2) Supervised Classification

<img src="https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/63621/versions/2/screenshot.gif" width="220" height="150">

3) Unsupervised Regression    

4) Unsupervised Classification

## Dataclasses

To get the ball rolling, I will start with numpy arrays as a base data class.  I may build on top of these, monkeypatching specific functionality.  At a later time I will attempt to build Vector and Matrix data classes from scratch in base python.  

## Design Assumptions

* All inputs will be lists or numpy arrays
