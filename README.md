# activeSVC
ActiveSVC selects features for large matrix data with reduced computational complexity or limited data acquisition. It approaches Sequential Feature Selection through an active learning strategy with a support vector machine classifier. At each round of iteration, the procedure analyzes only the samples that classify poorly with the current feature set, and the procedure extends the feature set by identifying features within incorrectly classified samples that will maximally shift the classification margin. There are two strategy, min_complexity and min_acquisition. Min_complexity strategy tends to use less samples each iteration while min_acquisition strategy tends to re-use samples used in previous iterations to minimize the total samples we acquired during the procedure.

## Why is activeSVC better than other feature selection methods?
- Easy to use
- Good for large datasets
- Reduce computational complexity
- Minimize the data size we need

## Usage
ActiveSVC processes a datasets with training set and test set and returns the features selected, training accuracy, test accuracy, training mean squared error, test mean squared error, the number of samples acquired after every features are selected.

## Requires
numpy, random, math, os, time, multiprocessing, sklearn, matplotlib

## Import
    from activeSVC import min_complexity
    from activeSVC import min_acquisition

## Function
- min_complexity
- min_acquisition

## min_complexity
### Parameters
    X_train: {ndarray, sparse matrix} of shape {n_samples_X, n_features}
            Input data of training set.
    y_train: ndarray of shape {n_samples_X,}
            Input classification labels of training set.
    X_test: {ndarray, sparse matrix} of shape {n_samples_X, n_features}
            Input data of test set.
    y_test: ndarray of shape {n_samples_X,}
            Input classification labels of test set.
    num_features: integer
            The total number of features to select.
    num_samples: integer
            The number of samples to use in each iteration (for each feature).
    init_samples: integer, default=None
            The number of samples to use in the first iteration.
    balance: bool, default=False
            Balance samples of each classes when sampling misclassified samples at each iteration or randomly sample misclassified samples.

### Return
    feature_selected: list of integer
            The sequence of features selected.
    num_samples_list: list of integer
            The number of unique samples acquired totally after every features are selected.
    train_errors: list of float
            Mean squared error of training set after every features are selected.
    test_errors: list of float
            Mean squared error of test set after every features are selected.
    train_accuracy: list of float
            Classification accuracy of training set after every features are selected.
    test_accuracy: list of float
            Classification accuracy of test set after every features are selected.

## min_acquisition
### Parameters
    X_train: {ndarray, sparse matrix} of shape {n_samples_X, n_features}
            Input data of training set.
    y_train: ndarray of shape {n_samples_X,}
            Input classification labels of training set.
    X_test: {ndarray, sparse matrix} of shape {n_samples_X, n_features}
            Input data of test set.
    y_test: ndarray of shape {n_samples_X,}
            Input classification labels of test set.
    num_features: integer
            The total number of features to select.
    num_samples: integer
            The number of misclassified samples randomly sampled, which are taken union with samples already acquired before. The union of samples are used for next ietration.
    init_samples: integer, default=None
            The number of samples to use in the first iteration.

### Return
    feature_selected: list of integer
            The sequence of features selected.
    num_samples_list: list of integer
            The number of unique samples acquired totally after every features are selected.
    samples_global: list of integer
            The indices of samples that are acquired.
    train_errors: list of float
            Mean squared error of training set after every features are selected.
    test_errors: list of float
            Mean squared error of test set after every features are selected.
    train_accuracy: list of float
            Classification accuracy of training set after every features are selected.
    test_accuracy: list of float
            Classification accuracy of test set after every features are selected.

