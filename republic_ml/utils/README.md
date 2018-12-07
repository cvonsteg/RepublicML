# Republic Utilities

The utilities (commonly `utils`) contain the building blocks, and base functionality which make up the `Republic ML` data model.

## Cross Validator

The `CrossValidator` base class allows for statistical cross validation functionality to be inherited by each model class.  This functionality entails splitting the dataset (in accordance with a proivded split ratio) allowing for part of the data to be used to train and fit the model (`Training Split`), and the remaining part to be used to test and validate the model (`Test Split`) from each dataset.

