import numpy as np
import random
import math
import os
import time
from parfor import parfor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize 
from sklearn.metrics.pairwise import cosine_similarity


class TimerError(Exception):
     """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        

        

def text_create(path, name, msg):
    full_path = path + "/" + name + '.txt'
    file = open(full_path, 'w')
    file.write(str(msg))



def SVM(X, y):
    model = svm.LinearSVC(max_iter=1000000)
    model.fit(X, y)
    return model


def get_error(model, X, y):
    y_pred = model.predict(X)
    return mean_squared_error(y_pred, y)


def select_samples_mincomplexity(X, y, num_samples,balance=False):
    model = SVM(X, y)
    y_pred = model.predict(X)
    sv = [i for i in range(len(y)) if y[i] != y_pred[i]]
    if balance:
        indices=[]
        classes=np.unique(y)
        @parfor(classes,bar=False)
        def sv_classes(c):
            sv_class = list(set(list(np.where(y == c)[0])) & set(sv))
            return sv_class
        sv_classes.sort(key=len)
        for i in range(len(classes)):
            sv_class=sv_classes[i]
            at_least=int((num_samples-len(indices))/(len(classes)-i))
            if len(sv_class)<=at_least:
                indices+=sv_class
            else:
                indices += random.sample(sv_class, at_least)
    else:
        if len(sv)<num_samples:
            indices =sv
        else:
            indices = random.sample(sv, num_samples)
    return indices, model


def select_samples_minacquisition(X, y, num_samples, sample_selected):
    model = SVM(X, y)
    y_pred = model.predict(X)
    sv = [i for i in range(len(y)) if y[i] != y_pred[i]]
    reused = list(set(sample_selected) & set(sv))
    num_select=num_samples-len(reused)
    if num_select<=0:
        return [],model
    else:
        indices = reused
        sv = list(set(sv) - set(indices))
        if len(sv)<=num_select:
            indices +=sv
        else:
            indices += random.sample(sv, num_select)
    return indices, model




def select_feature(X, y, feature_list):
    coef_ = SVM(X[:, feature_list], y).coef_
    w_padded = np.hstack((coef_, np.zeros((coef_.shape[0], 1))))
    
    
    @parfor(range(X.shape[1]),bar=False)
    def angles(i):
        X_local = X[:, feature_list + [i]]
        w_new = SVM(X_local, y).coef_
        cos=cosine_similarity(w_padded, w_new)
        angle = 0
        for j in range(w_padded.shape[0]):
            tmp=cos[j,j]
            if tmp>1:
                tmp=1
            elif tmp<-1:
                tmp=-1
            angle = angle + math.acos(tmp)            
        return angle
    indices = sorted(range(X.shape[1]), key=lambda i: angles[i], reverse=True)
    return [i for i in indices if i not in feature_list][0]


def min_complexity(X_train, y_train, X_test, y_test, num_features, num_samples,balance=False):
    feature_selected = []
    num_samples_list = []
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []
    
    if balance:
        samples=[]
        classes=np.unique(y_train)
        sample_classes=[]
        for c in classes:
            sample_class = list(np.where(y_train == c)[0])
            sample_classes.append(sample_class)
        sample_classes.sort(key=len)
        for i in range(len(classes)):
            sample_class=sample_classes[i]
            at_least=int((num_samples-len(samples))/(len(classes)-i))
            if len(sample_class)<=at_least:
                samples+=sample_class
            else:
                samples += random.sample(sample_class, at_least)
    else:
        shuffle = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle)
        samples = shuffle[:num_samples]
            
    X_global = X_train[samples, :]
    y_global = y_train[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))

    @parfor(range(X_global.shape[1]),bar=False)
    def scores(i):
        model=SVM(X_global[:,i].reshape(-1, 1),y_global)
        return model.score(X_global[:,i].reshape(-1, 1),y_global)  # R^2 for regression and mean accuracy for classificarion

    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[0]
    feature_selected.append(new_feature)

    for i in range(num_features - 1):
        t=Timer()
        t.start()

        X_measured_train = X_train[:,feature_selected]
        X_measured_test = X_test[:,feature_selected]

        samples, model = select_samples_mincomplexity(X_measured_train, y_train, num_samples,balance=balance)
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))

        train_error = get_error(model, X_measured_train, y_train)
        test_error = get_error(model, X_measured_test, y_test)
        train_score = model.score(X_measured_train, y_train)
        test_score = model.score(X_measured_test, y_test)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(i) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

        new_feature=select_feature(X_train[samples], y_train[samples],feature_selected)
        feature_selected.append(new_feature)
        t.stop()

    X_measured_train = X_train[:,feature_selected]
    X_measured_test = X_test[:,feature_selected]
    model=SVM(X_measured_train,y_train)
    train_error = get_error(model, X_measured_train, y_train)
    test_error = get_error(model, X_measured_test, y_test)
    train_score = model.score(X_measured_train, y_train)
    test_score = model.score(X_measured_test, y_test)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(i+1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

    return feature_selected, num_samples_list, train_errors, test_errors, train_scores, test_scores





def min_acquisition(X_train, y_train, X_test, y_test, num_features, num_samples):
    feature_selected = []
    num_samples_list = []
    samples_global=[]
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []

    shuffle = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle)
    samples = shuffle[:num_samples]
    X_global = X_train[samples, :]
    y_global = y_train[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))

    @parfor(range(X_global.shape[1]),bar=False)
    def scores(i):
        model=SVM(X_global[:,i].reshape(-1, 1),y_global)
        return model.score(X_global[:,i].reshape(-1, 1),y_global)  # R^2 for regression and mean accuracy for classificarion

    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[0]
    feature_selected.append(new_feature)

    for i in range(num_features - 1):
        t=Timer()
        t.start()

        X_measured_train = X_train[:,feature_selected]
        X_measured_test = X_test[:,feature_selected]

        samples, model = select_samples_minacquisition(X_measured_train, y_train, num_samples,samples_global)
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))

        train_error = get_error(model, X_measured_train, y_train)
        test_error = get_error(model, X_measured_test, y_test)
        train_score = model.score(X_measured_train, y_train)
        test_score = model.score(X_measured_test, y_test)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(i) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

        new_feature=select_feature(X_train[samples_global], y_train[samples_global],feature_selected)
        feature_selected.append(new_feature)
        t.stop()

    X_measured_train = X_train[:,feature_selected]
    X_measured_test = X_test[:,feature_selected]
    model=SVM(X_measured_train,y_train)
    train_error = get_error(model, X_measured_train, y_train)
    test_error = get_error(model, X_measured_test, y_test)
    train_score = model.score(X_measured_train, y_train)
    test_score = model.score(X_measured_test, y_test)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(i+1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

    return feature_selected, num_samples_list, samples_global, train_errors, test_errors, train_scores, test_scores