import numpy as np
import random
import math
import os
import time
import multiprocessing as mp
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold,GridSearchCV
import scipy.sparse as sp_sparse
from scipy.sparse import vstack, hstack

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
        return elapsed_time
        

        

def text_create(path, name, msg):
    full_path = path + "/" + name + '.pickle'
    f=open(full_path,'wb') 
    pickle.dump(msg,f)
    f.close()



def SVM(X, y, penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model = svm.LinearSVC(penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,  
                          random_state=random_state, max_iter=max_iter)
    model.fit(X, y)
    return model


def get_error(model, X, y,sample_weight):
    y_pred = model.predict(X)
    return mean_squared_error(y_pred, y,sample_weight=sample_weight)


def get_sv_classes(c,y,sv):
    sv_classes = list(set(list(np.where(y == c)[0])) & set(sv))
    return sv_classes
def select_samples_mincomplexity(X, y, num_samples,balance=False,penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model = SVM(X, y,penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight, 
                          random_state=random_state, max_iter=max_iter)
    y_pred = model.predict(X)
    sv = [i for i in range(len(y)) if y[i] != y_pred[i]]
    if balance:
        indices=[]
        classes=np.unique(y)
        
        pool = mp.Pool(mp.cpu_count())
        sv_classes=pool.starmap(get_sv_classes,[(c,y,sv) for c in classes])
        pool.close()
        
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


def select_samples_minacquisition(X, y, num_samples, sample_selected,balance=False,penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model = SVM(X, y,penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight, 
                          random_state=random_state, max_iter=max_iter)
    y_pred = model.predict(X)
    sv = [i for i in range(len(y)) if y[i] != y_pred[i]]
    reused = list(set(sample_selected) & set(sv))
    num_select=num_samples-len(reused)
    if num_select<=0:
        return [],model
    elif balance:
        indices = reused
        sv = list(set(sv) - set(indices))
        classes=np.unique(y)
        
        pool = mp.Pool(mp.cpu_count())
        sv_classes=pool.starmap(get_sv_classes,[(c,y,sv) for c in classes])
        pool.close()
        
        sv_classes.sort(key=len)
        for i in range(len(classes)):
            sv_class=sv_classes[i]
            at_least=int((num_samples-len(indices))/(len(classes)-i))
            if len(sv_class)<=at_least:
                indices+=sv_class
            else:
                indices += random.sample(sv_class, at_least)
    else:
        indices = reused
        sv = list(set(sv) - set(indices))
        if len(sv)<=num_select:
            indices +=sv
        else:
            indices += random.sample(sv, num_select)
    return indices, model




def get_angles(i, X, y, feature_list,w_padded,penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    X_local = X[:, feature_list + [i]]
    w_new = SVM(X_local, y,penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter).coef_
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
def select_feature(X, y, feature_list,penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    coef_ = SVM(X[:, feature_list], y,penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter).coef_
    w_padded = np.hstack((coef_, np.zeros((coef_.shape[0], 1))))
    
    pool = mp.Pool(mp.cpu_count())
    angles=pool.starmap(get_angles, [(i, X, y,feature_list,w_padded,penalty,loss,dual, tol, C, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X.shape[1])])
    pool.close()

    indices = sorted(range(X.shape[1]), key=lambda i: angles[i], reverse=True)
    return [i for i in indices if i not in feature_list][0]



def get_scores(i, X_global, y_global,penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model=SVM(X_global[:,i].reshape(-1, 1),y_global,penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y_global,return_inverse=True, return_counts=True)
        sample_weight=(y_global.shape[0]/(len(classes)*count))[inverse]
    else:
        sample_weight=None
    return model.score(X_global[:,i].reshape(-1, 1),y_global, sample_weight=sample_weight)
def min_complexity(X_train, y_train, X_test, y_test, num_features, num_samples,init_features=1,init_samples=None, balance=False,
                   penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    feature_selected = []
    num_samples_list = []
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []
    step_times=[]
    if init_samples is None:
        init_samples=num_samples
    
    t=Timer()
    t.start()
    
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
            at_least=int((init_samples-len(samples))/(len(classes)-i))
            if len(sample_class)<=at_least:
                samples+=sample_class
            else:
                samples += random.sample(sample_class, at_least)
    else:
        shuffle = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle)
        samples = shuffle[:init_samples]
            
    X_global = X_train[samples, :]
    y_global = y_train[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))

    pool = mp.Pool(mp.cpu_count())
    scores=pool.starmap(get_scores, [(i,X_global, y_global,penalty,loss,dual, tol, C, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X_global.shape[1])])
    pool.close() 
    
    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[:init_features]
    feature_selected=new_feature
    
    step_times.append(t.stop())
    
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y_train,return_inverse=True, return_counts=True)
        train_sample_weight=(y_train.shape[0]/(len(classes)*count))[inverse]
        classes, inverse, count=np.unique(y_test,return_inverse=True, return_counts=True)
        test_sample_weight=(y_test.shape[0]/(len(classes)*count))[inverse]
    else:
        train_sample_weight=None
        test_sample_weight=None
        
    for i in range(num_features - 1):
        t=Timer()
        t.start()

        X_measured_train = X_train[:,feature_selected]
        X_measured_test = X_test[:,feature_selected]

        samples, model = select_samples_mincomplexity(X_measured_train, y_train, num_samples,balance=balance,
                                                     penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                                      intercept_scaling=intercept_scaling, class_weight=class_weight,
                                                      random_state=random_state, max_iter=max_iter)

        train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
        test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
        train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
        test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(i) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))
        
        new_feature=select_feature(X_train[samples], y_train[samples],feature_selected,
                                   penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
        feature_selected.append(new_feature)
        step_times.append(t.stop())

    X_measured_train = X_train[:,feature_selected]
    X_measured_test = X_test[:,feature_selected]
    model=SVM(X_measured_train,y_train,penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
    test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
    train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
    test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(i+1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

    return feature_selected, num_samples_list, train_errors, test_errors, train_scores, test_scores,step_times




def min_acquisition(X_train, y_train, X_test, y_test, num_features, num_samples,init_features=1,init_samples=None,balance=False,
                    penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    feature_selected = []
    num_samples_list = []
    samples_global=[]
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []
    step_times=[]
    
    if init_samples is None:
        init_samples=num_samples
    
    t=Timer()
    t.start()
    
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
            at_least=int((init_samples-len(samples))/(len(classes)-i))
            if len(sample_class)<=at_least:
                samples+=sample_class
            else:
                samples += random.sample(sample_class, at_least)
    else:
        shuffle = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle)
        samples = shuffle[:init_samples]
        
    X_global = X_train[samples, :]
    y_global = y_train[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))

    pool = mp.Pool(mp.cpu_count())
    scores=pool.starmap(get_scores, [(i,X_global, y_global,penalty,loss,dual, tol, C, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X_global.shape[1])])
    pool.close() 

    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[:init_features]
    feature_selected=new_feature
    
    step_times.append(t.stop())
    
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y_train,return_inverse=True, return_counts=True)
        train_sample_weight=(y_train.shape[0]/(len(classes)*count))[inverse]
        classes, inverse, count=np.unique(y_test,return_inverse=True, return_counts=True)
        test_sample_weight=(y_test.shape[0]/(len(classes)*count))[inverse]
    else:
        train_sample_weight=None
        test_sample_weight=None
        
    for i in range(num_features - 1):
        t=Timer()
        t.start()

        X_measured_train = X_train[:,feature_selected]
        X_measured_test = X_test[:,feature_selected]

        samples, model = select_samples_minacquisition(X_measured_train, y_train, num_samples,samples_global,balance=balance,
                                                       penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)

        train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
        test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
        train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
        test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(i) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))
        
        new_feature=select_feature(X_train[samples_global], y_train[samples_global],feature_selected,
                                   penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
        feature_selected.append(new_feature)
        step_times.append(t.stop())

    X_measured_train = X_train[:,feature_selected]
    X_measured_test = X_test[:,feature_selected]
    model=SVM(X_measured_train,y_train,penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
    test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
    train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
    test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(i+1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

    return feature_selected, num_samples_list, samples_global, train_errors, test_errors, train_scores, test_scores,step_times


def SVM_parallel(X, y, tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = svm.LinearSVC(penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,  
                          random_state=random_state, max_iter=max_iter)
    
    space = dict()
    space['tol'] = tol
    space['C'] = C
    search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv, refit=True)
    search.fit(X,y)
    return search.best_estimator_

def SVM_parallel_para(X, y, tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000): 
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = svm.LinearSVC(penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,  
                          random_state=random_state, max_iter=max_iter)
    
    space = dict()
    space['tol'] = tol
    space['C'] = C
    search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv, refit=True)
    search.fit(X,y)
    return search.best_estimator_, search.best_params_

def SVM_cv(X, y, tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = svm.LinearSVC(penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,  
                          random_state=random_state, max_iter=max_iter)
    space = dict()
    space['tol'] = tol
    space['C'] = C
    search = GridSearchCV(model, space, scoring='accuracy', n_jobs=1, cv=cv, refit=True)
    search.fit(X,y)
    return search.best_estimator_

def select_samples_mincomplexity_cv(X, y, num_samples,balance=False,tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model,para = SVM_parallel_para(X, y,tol=tol, C=C, n_splits=n_splits, penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight, 
                          random_state=random_state, max_iter=max_iter)
    y_pred = model.predict(X)
    sv = [i for i in range(len(y)) if y[i] != y_pred[i]]
    if balance:
        indices=[]
        classes=np.unique(y)
        
        pool = mp.Pool(mp.cpu_count())
        sv_classes=pool.starmap(get_sv_classes,[(c,y,sv) for c in classes])
        pool.close()
        
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
    return indices, model,para


def select_samples_minacquisition_cv(X, y, num_samples, sample_selected,balance=False,tol=[1e-4], C=[1], n_splits=5,penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model,para = SVM_parallel_para(X, y,tol=tol, C=C, n_splits=n_splits, penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight, 
                          random_state=random_state, max_iter=max_iter)
    y_pred = model.predict(X)
    sv = [i for i in range(len(y)) if y[i] != y_pred[i]]
    reused = list(set(sample_selected) & set(sv))
    num_select=num_samples-len(reused)
    if num_select<=0:
        return [],model,para
    elif balance:
        indices = reused
        sv = list(set(sv) - set(indices))
        classes=np.unique(y)
        
        pool = mp.Pool(mp.cpu_count())
        sv_classes=pool.starmap(get_sv_classes,[(c,y,sv) for c in classes])
        pool.close()
        
        sv_classes.sort(key=len)
        for i in range(len(classes)):
            sv_class=sv_classes[i]
            at_least=int((num_samples-len(indices))/(len(classes)-i))
            if len(sv_class)<=at_least:
                indices+=sv_class
            else:
                indices += random.sample(sv_class, at_least)
    else:
        indices = reused
        sv = list(set(sv) - set(indices))
        if len(sv)<=num_select:
            indices +=sv
        else:
            indices += random.sample(sv, num_select)
    return indices, model,para




def get_angles_cv(i, X, y, feature_list,w_padded,tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    X_local = X[:, feature_list + [i]]
    w_new = SVM_cv(X_local, y,tol=tol, C=C, n_splits=n_splits, penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter).coef_
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
def select_feature_cv(X, y, feature_list,tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model,para=SVM_parallel_para(X[:, feature_list], y,tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    coef_ = model.coef_
    w_padded = np.hstack((coef_, np.zeros((coef_.shape[0], 1))))
    
    pool = mp.Pool(mp.cpu_count())
    angles=pool.starmap(get_angles_cv, [(i, X, y,feature_list,w_padded,tol,C,n_splits,penalty,loss,dual, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X.shape[1])])
    pool.close()

    indices = sorted(range(X.shape[1]), key=lambda i: angles[i], reverse=True)
    return [i for i in indices if i not in feature_list][0],para



def get_scores_cv(i, X_global, y_global,tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    model=SVM_cv(X_global[:,i].reshape(-1, 1),y_global,tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual,fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y_global,return_inverse=True, return_counts=True)
        sample_weight=(y_global.shape[0]/(len(classes)*count))[inverse]
    else:
        sample_weight=None
    return model.score(X_global[:,i].reshape(-1, 1),y_global, sample_weight=sample_weight)
def min_complexity_cv(X_train, y_train, X_test, y_test, num_features, num_samples,init_features=1, init_samples=None, balance=False,
                   tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    num_samples_list = []
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []
    paras=[]
    step_times=[]
    if init_samples is None:
        init_samples=num_samples
    
    t=Timer()
    t.start()
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
            at_least=int((init_samples-len(samples))/(len(classes)-i))
            if len(sample_class)<=at_least:
                samples+=sample_class
            else:
                samples += random.sample(sample_class, at_least)
    else:
        shuffle = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle)
        samples = shuffle[:init_samples]
            
    X_global = X_train[samples, :]
    y_global = y_train[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))
    

    pool = mp.Pool(mp.cpu_count())
    scores=pool.starmap(get_scores_cv, [(i,X_global, y_global,tol,C,n_splits,penalty,loss,dual, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X_global.shape[1])])
    pool.close() 
    
    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[:init_features]
    feature_selected=new_feature
    step_times.append(t.stop())
    
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y_train,return_inverse=True, return_counts=True)
        train_sample_weight=(y_train.shape[0]/(len(classes)*count))[inverse]
        classes, inverse, count=np.unique(y_test,return_inverse=True, return_counts=True)
        test_sample_weight=(y_test.shape[0]/(len(classes)*count))[inverse]
    else:
        train_sample_weight=None
        test_sample_weight=None
    
    for i in range(num_features - init_features):
        t=Timer()
        t.start()

        X_measured_train = X_train[:,feature_selected]
        X_measured_test = X_test[:,feature_selected]
        
        samples, model,para = select_samples_mincomplexity_cv(X_measured_train, y_train, num_samples,balance=balance,
                                                     tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                                                      intercept_scaling=intercept_scaling, class_weight=class_weight,
                                                      random_state=random_state, max_iter=max_iter)
        paras.append(para)
        
        train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
        test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
        train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
        test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(init_features+i) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))
        
        new_feature,para=select_feature_cv(X_train[samples], y_train[samples],feature_selected,
                                   tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
        feature_selected.append(new_feature)
        paras.append(para)
        step_times.append(t.stop())

    X_measured_train = X_train[:,feature_selected]
    X_measured_test = X_test[:,feature_selected]
    model=SVM_parallel(X_measured_train,y_train,tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
    test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
    train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
    test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(num_features) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

    return feature_selected, num_samples_list, train_errors, test_errors, train_scores, test_scores,paras,step_times




def min_acquisition_cv(X_train, y_train, X_test, y_test, num_features, num_samples,init_features=1,init_samples=None,balance=False,
                    tol=[1e-4], C=[1], n_splits=5, penalty='l2',loss='squared_hinge',dual=True, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    feature_selected = []
    num_samples_list = []
    samples_global=[]
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []
    paras=[]
    step_times=[]
    
    if init_samples is None:
        init_samples=num_samples
        
    t=Timer()
    t.start()
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
            at_least=int((init_samples-len(samples))/(len(classes)-i))
            if len(sample_class)<=at_least:
                samples+=sample_class
            else:
                samples += random.sample(sample_class, at_least)
    else:
        shuffle = np.arange(X_train.shape[0])
        np.random.shuffle(shuffle)
        samples = shuffle[:init_samples]
        
    X_global = X_train[samples, :]
    y_global = y_train[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))

    pool = mp.Pool(mp.cpu_count())
    scores=pool.starmap(get_scores_cv, [(i,X_global, y_global,tol,C,n_splits,penalty,loss,dual, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X_global.shape[1])])
    pool.close() 

    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[:init_features]
    feature_selected=new_feature
    step_times.append(t.stop())

        
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y_train,return_inverse=True, return_counts=True)
        train_sample_weight=(y_train.shape[0]/(len(classes)*count))[inverse]
        classes, inverse, count=np.unique(y_test,return_inverse=True, return_counts=True)
        test_sample_weight=(y_test.shape[0]/(len(classes)*count))[inverse]
    else:
        train_sample_weight=None
        test_sample_weight=None
        
    for i in range(num_features - 1):
        t=Timer()
        t.start()

        X_measured_train = X_train[:,feature_selected]
        X_measured_test = X_test[:,feature_selected]

        samples, model,para = select_samples_minacquisition_cv(X_measured_train, y_train, num_samples,samples_global,balance=balance,
                                                       tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual,fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
        paras.append(para)
        
        train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
        test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
        train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
        test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(i) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))
        
        new_feature,para=select_feature_cv(X_train[samples_global], y_train[samples_global],feature_selected,
                                   tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
        feature_selected.append(new_feature)
        paras.append(para)
        step_times.append(t.stop())

    X_measured_train = X_train[:,feature_selected]
    X_measured_test = X_test[:,feature_selected]
    model=SVM_parallel(X_measured_train,y_train,tol=tol, C=C, n_splits=n_splits,penalty=penalty,loss=loss,dual=dual,fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    train_error = get_error(model, X_measured_train, y_train,sample_weight=train_sample_weight)
    test_error = get_error(model, X_measured_test, y_test,sample_weight=test_sample_weight)
    train_score = model.score(X_measured_train, y_train,sample_weight=train_sample_weight)
    test_score = model.score(X_measured_test, y_test,sample_weight=test_sample_weight)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(i+1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

    return feature_selected, num_samples_list, samples_global, train_errors, test_errors, train_scores, test_scores,paras,step_times


def index_cell(data, indices, indptr, shape,index):
    i=index[0]
    matrix = sp_sparse.csc_matrix((data[indptr[i]:indptr[i+1]], indices[indptr[i]:indptr[i+1]],[0,indptr[i+1]-indptr[i]]),shape=(shape[0],1))
    for i in index[1:]:
        tmp=sp_sparse.csc_matrix((data[indptr[i]:indptr[i+1]], indices[indptr[i]:indptr[i+1]],[0,indptr[i+1]-indptr[i]]),shape=(shape[0],1))
        matrix=hstack((matrix,tmp))
    return matrix.transpose().tocsr()

def index_gene(data, indices, indptr, shape,index):
    i=index[0]
    matrix = sp_sparse.csr_matrix((data[indptr[i]:indptr[i+1]], indices[indptr[i]:indptr[i+1]],[0,indptr[i+1]-indptr[i]]),shape=(1,shape[1]))
    for i in index[1:]:
        tmp=sp_sparse.csr_matrix((data[indptr[i]:indptr[i+1]], indices[indptr[i]:indptr[i+1]],[0,indptr[i+1]-indptr[i]]),shape=(1,shape[1]))
        matrix=vstack((matrix,tmp))
    return matrix.transpose()


def min_complexity_h5py(data_cell,indices_cell,indptr_cell,data_gene,indices_gene,indptr_gene, y, shape,idx_train,idx_test,
                   num_features, num_samples,init_features=1,init_samples=None, balance=False,
                   penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    feature_selected = []
    num_samples_list = []
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []
    step_times=[]
    if init_samples is None:
        init_samples=num_samples
    
    t=Timer()
    t.start()
    
    if balance:
        samples=[]
        classes=np.unique(y[idx_train])
        sample_classes=[]
        for c in classes:
            sample_class = list(np.where(y[idx_train] == c)[0])
            sample_classes.append(sample_class)
        sample_classes.sort(key=len)
        for i in range(len(classes)):
            sample_class=sample_classes[i]
            at_least=int((init_samples-len(samples))/(len(classes)-i))
            if len(sample_class)<=at_least:
                samples+=sample_class
            else:
                samples += random.sample(sample_class, at_least)
    else:
        shuffle = np.array(idx_train)
        np.random.shuffle(shuffle)
        samples = shuffle[:init_samples]
            
    X_global = index_cell(data_cell,indices_cell,indptr_cell,shape,samples)
    y_global = y[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))

    pool = mp.Pool(mp.cpu_count())
    scores=pool.starmap(get_scores, [(i,X_global, y_global,penalty,loss,dual, tol, C, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X_global.shape[1])])
    pool.close() 
    
    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[:init_features]
    feature_selected=new_feature
    
    step_times.append(t.stop())
    
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y[idx_train],return_inverse=True, return_counts=True)
        train_sample_weight=(len(idx_train)/(len(classes)*count))[inverse]
        classes, inverse, count=np.unique(y[idx_test],return_inverse=True, return_counts=True)
        test_sample_weight=(len(idx_test)/(len(classes)*count))[inverse]
    else:
        train_sample_weight=None
        test_sample_weight=None
        
    for i in range(num_features - init_features):
        t=Timer()
        t.start()

        X_measured = index_gene(data_gene,indices_gene,indptr_gene, shape,feature_selected)
        X_measured_train = X_measured[idx_train,:]
        X_measured_test = X_measured[idx_test,:]
        del X_measured

        raw_samples, model = select_samples_mincomplexity(X_measured_train, y[idx_train], num_samples,balance=balance,
                                                     penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                                      intercept_scaling=intercept_scaling, class_weight=class_weight,
                                                      random_state=random_state, max_iter=max_iter)
        samples=np.array(idx_train)[raw_samples].tolist()
        
        train_error = get_error(model, X_measured_train, y[idx_train],sample_weight=train_sample_weight)
        test_error = get_error(model, X_measured_test, y[idx_test],sample_weight=test_sample_weight)
        train_score = model.score(X_measured_train, y[idx_train],sample_weight=train_sample_weight)
        test_score = model.score(X_measured_test, y[idx_test],sample_weight=test_sample_weight)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(init_features+i-1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))
        
        X_global = index_cell(data_cell,indices_cell,indptr_cell,shape,samples)
        y_global = y[samples]
        new_feature=select_feature(X_global, y_global,feature_selected,
                                   penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
        feature_selected.append(new_feature)
        step_times.append(t.stop())

    X_measured = index_gene(data_gene,indices_gene,indptr_gene, shape,feature_selected)
    X_measured_train = X_measured[idx_train,:]
    X_measured_test = X_measured[idx_test,:]
    del X_measured
    model=SVM(X_measured_train,y[idx_train],penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    train_error = get_error(model, X_measured_train, y[idx_train],sample_weight=train_sample_weight)
    test_error = get_error(model, X_measured_test, y[idx_test],sample_weight=test_sample_weight)
    train_score = model.score(X_measured_train, y[idx_train],sample_weight=train_sample_weight)
    test_score = model.score(X_measured_test, y[idx_test],sample_weight=test_sample_weight)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(num_features-1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))

    return feature_selected, num_samples_list, train_errors, test_errors, train_scores, test_scores,step_times




def min_acquisition_h5py(data_cell,indices_cell,indptr_cell,data_gene,indices_gene,indptr_gene, y, shape,idx_train,idx_test,
                    num_features, num_samples,init_features=1,init_samples=None,balance=False,
                    penalty='l2',loss='squared_hinge',dual=True, tol=1e-4, C=1.0, fit_intercept=True,
                          intercept_scaling=1, class_weight=None, random_state=None, max_iter=1000):
    feature_selected = []
    num_samples_list = []
    train_errors = []
    test_errors = []
    train_scores = []
    test_scores = []
    step_times=[]
    if init_samples is None:
        init_samples=num_samples
    
    t=Timer()
    t.start()
    
    if balance:
        samples=[]
        classes=np.unique(y[idx_train])
        sample_classes=[]
        for c in classes:
            sample_class = list(np.where(y[idx_train] == c)[0])
            sample_classes.append(sample_class)
        sample_classes.sort(key=len)
        for i in range(len(classes)):
            sample_class=sample_classes[i]
            at_least=int((init_samples-len(samples))/(len(classes)-i))
            if len(sample_class)<=at_least:
                samples+=sample_class
            else:
                samples += random.sample(sample_class, at_least)
    else:
        shuffle = np.array(idx_train)
        np.random.shuffle(shuffle)
        samples = shuffle[:init_samples]
            
    X_global = index_cell(data_cell,indices_cell,indptr_cell,shape,samples)
    y_global = y[samples]
    samples_global=samples
    num_samples_list.append(len(samples_global))

    pool = mp.Pool(mp.cpu_count())
    scores=pool.starmap(get_scores, [(i,X_global, y_global,penalty,loss,dual, tol, C, fit_intercept,
                                      intercept_scaling, class_weight,
                                      random_state, max_iter) for i in range(X_global.shape[1])])
    pool.close() 
    
    new_feature = sorted(range(X_global.shape[1]), key=lambda i: scores[i], reverse=True)[:init_features]
    feature_selected=new_feature
    
    step_times.append(t.stop())
    
    if class_weight=='balanced':
        classes, inverse, count=np.unique(y[idx_train],return_inverse=True, return_counts=True)
        train_sample_weight=(len(idx_train)/(len(classes)*count))[inverse]
        classes, inverse, count=np.unique(y[idx_test],return_inverse=True, return_counts=True)
        test_sample_weight=(len(idx_test)/(len(classes)*count))[inverse]
    else:
        train_sample_weight=None
        test_sample_weight=None
        
    for i in range(num_features - init_features):
        t=Timer()
        t.start()

        X_measured = index_gene(data_gene,indices_gene,indptr_gene, shape,feature_selected)
        X_measured_train = X_measured[idx_train,:]
        X_measured_test = X_measured[idx_test,:]
        del X_measured

        raw_samples, model = select_samples_minacquisition(X_measured_train, y[idx_train], num_samples,samples_global, balance=balance,
                                                     penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                                      intercept_scaling=intercept_scaling, class_weight=class_weight,
                                                      random_state=random_state, max_iter=max_iter)
        samples=np.array(idx_train)[raw_samples].tolist()
        
        train_error = get_error(model, X_measured_train, y[idx_train],sample_weight=train_sample_weight)
        test_error = get_error(model, X_measured_test, y[idx_test],sample_weight=test_sample_weight)
        train_score = model.score(X_measured_train, y[idx_train],sample_weight=train_sample_weight)
        test_score = model.score(X_measured_test, y[idx_test],sample_weight=test_sample_weight)
        train_errors.append(train_error)
        test_errors.append(test_error)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("feature " + str(init_features+i-1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
        print('training error=' + str(train_error) + ' test error=' + str(test_error))
        print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))
        samples_global = list(set().union(samples_global, samples))
        num_samples_list.append(len(samples_global))
        
        X_global = index_cell(data_cell,indices_cell,indptr_cell,shape,samples_global)
        y_global = y[samples_global]
        new_feature=select_feature(X_global, y_global,feature_selected,
                                   penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
        feature_selected.append(new_feature)
        step_times.append(t.stop())

    X_measured = index_gene(data_gene,indices_gene,indptr_gene, shape,feature_selected)
    X_measured_train = X_measured[idx_train,:]
    X_measured_test = X_measured[idx_test,:]
    del X_measured
    model=SVM(X_measured_train,y[idx_train],penalty=penalty,loss=loss,dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                          random_state=random_state, max_iter=max_iter)
    train_error = get_error(model, X_measured_train, y[idx_train],sample_weight=train_sample_weight)
    test_error = get_error(model, X_measured_test, y[idx_test],sample_weight=test_sample_weight)
    train_score = model.score(X_measured_train, y[idx_train],sample_weight=train_sample_weight)
    test_score = model.score(X_measured_test, y[idx_test],sample_weight=test_sample_weight)
    train_errors.append(train_error)
    test_errors.append(test_error)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print("feature " + str(num_features-1) + ' : gene ' + str(new_feature)+'  '+str(len(samples_global)) + ' samples')
    print('training error=' + str(train_error) + ' test error=' + str(test_error))
    print('training accuracy=' + str(train_score) + ' test accuracy=' + str(test_score))
    return feature_selected, num_samples_list, samples_global, train_errors, test_errors, train_scores, test_scores,step_times


