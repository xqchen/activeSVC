# activeSVC
ActiveSVC selects features for large matrix data with reduced computational complexity or limited data acquisition. It approaches Sequential Feature Selection through an active learning strategy with a support vector machine classifier. At each round of iteration, the procedure analyzes only the samples that classify poorly with the current feature set, and the procedure extends the feature set by identifying features within incorrectly classified samples that will maximally shift the classification margin. There are two strategy, min_complexity and min_acquisition. Min_complexity strategy tends to use less samples each iteration while min_acquisition strategy tends to re-use samples used in previous iterations to minimize the total samples we acquired during the procedure.

## Why is activeSVC better than other feature selection methods?
- Easy to use
- Good for large datasets
- Reduce computational complexity
- Minimize the data size we need

## Usage
ActiveSVC processes a datasets with training set and test set and returns the features selected, training accuracy, test accuracy, training mean squared error, test mean squared error, the number of samples acquired after every features are selected. We highly recommend to do l2-normalization for each sample before activeSVC to improve accuracy and speed up model training. 

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
            Balance samples of each classes when sampling misclassified samples at each 
            iteration or randomly sample misclassified samples.
    penalty: {‘l1’, ‘l2’}, default=’l2’
            Specifies the norm used in the penalization. The ‘l2’ penalty is the 
            standard used in SVC. The ‘l1’ leads to sparse weight.
    loss: {‘hinge’, ‘squared_hinge’}, default=’squared_hinge’
            Specifies the loss function for each SVC to train. ‘hinge’ is the standard 
            SVM loss while ‘squared_hinge’ is the square of the hinge loss. 
            The combination of penalty='l1' and loss='hinge' is not supported.
    dual: bool, default=True
            Select the algorithm to either solve the dual or primal optimization 
            problem for each SVC. Prefer dual=False when n_samples > n_features.
    tol: float, default=1e-4
            Tolerance for stopping criteria for each SVC.
    C: float, default=1.0
            Regularization parameter for each SVC. The strength of the regularization 
            is inversely proportional to C. Must be strictly positive.
    fit_intercept: bool, default=True
            Whether to calculate the intercept for each SVC. If set to false, no 
            intercept will be used in calculations (i.e. data is already centered).
    intercept_scaling: float, default=1
            When self.fit_intercept is True, instance vector x becomes 
            [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value 
            equals to intercept_scaling is appended to the instance vector. The intercept 
            becomes intercept_scaling * synthetic feature weight Note! the synthetic 
            feature weight is subject to l1/l2 regularization as all other features. 
            To lessen the effect of regularization on synthetic feature weight 
            (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight: dict or ‘balanced’, default=None
            Set the parameter C of class i to class_weight[i]*C for SVC. If not given, 
            all classes are supposed to have weight one. The “balanced” mode uses the 
            values of y to automatically adjust weights inversely proportional to class 
            frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
    random_state: int, RandomState instance or None, default=None
            Controls the pseudo random number generation for shuffling data for the dual 
            coordinate descent (if dual=True). When dual=False underlying implementation 
            of LinearSVC is not random and random_state has no effect on the results. 
            Pass an int for reproducible output across multiple function calls.
    max_iter: int, default=1000
            The maximum number of iterations to be run for each SVC.


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
            The number of misclassified samples randomly sampled, which are taken union with 
            samples already acquired before. The union of samples are used for next ietration.
    init_samples: integer, default=None
            The number of samples to use in the first iteration.
    penalty: {‘l1’, ‘l2’}, default=’l2’
            Specifies the norm used in the penalization. The ‘l2’ penalty is the 
            standard used in SVC. The ‘l1’ leads to sparse weight.
    loss: {‘hinge’, ‘squared_hinge’}, default=’squared_hinge’
            Specifies the loss function for each SVC to train. ‘hinge’ is the standard 
            SVM loss while ‘squared_hinge’ is the square of the hinge loss. 
            The combination of penalty='l1' and loss='hinge' is not supported.
    dual: bool, default=True
            Select the algorithm to either solve the dual or primal optimization 
            problem for each SVC. Prefer dual=False when n_samples > n_features.
    tol: float, default=1e-4
            Tolerance for stopping criteria for each SVC.
    C: float, default=1.0
            Regularization parameter for each SVC. The strength of the regularization 
            is inversely proportional to C. Must be strictly positive.
    fit_intercept: bool, default=True
            Whether to calculate the intercept for each SVC. If set to false, no 
            intercept will be used in calculations (i.e. data is already centered).
    intercept_scaling: float, default=1
            When self.fit_intercept is True, instance vector x becomes 
            [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value 
            equals to intercept_scaling is appended to the instance vector. The intercept 
            becomes intercept_scaling * synthetic feature weight Note! the synthetic 
            feature weight is subject to l1/l2 regularization as all other features. 
            To lessen the effect of regularization on synthetic feature weight 
            (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight: dict or ‘balanced’, default=None
            Set the parameter C of class i to class_weight[i]*C for SVC. If not given, 
            all classes are supposed to have weight one. The “balanced” mode uses the 
            values of y to automatically adjust weights inversely proportional to class 
            frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
    random_state: int, RandomState instance or None, default=None
            Controls the pseudo random number generation for shuffling data for the dual 
            coordinate descent (if dual=True). When dual=False underlying implementation 
            of LinearSVC is not random and random_state has no effect on the results. 
            Pass an int for reproducible output across multiple function calls.
    max_iter: int, default=1000
            The maximum number of iterations to be run for each SVC.

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

