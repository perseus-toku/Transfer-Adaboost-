import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_gaussian_quantiles
import sklearn.svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from scipy import sparse

from Tradaboost import Tradaboost




def  main():
    sci_categories = [ "sci.crypt","sci.electronics", "sci.med", "sci.space"]
    rec_categories = ["rec.autos",  "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"]

    rec_train = fetch_20newsgroups(subset='train',categories = rec_categories, shuffle = False, random_state = 42)
    rec_test = fetch_20newsgroups(subset='test',categories = rec_categories, shuffle = False, random_state = 42)

    sci_train = fetch_20newsgroups(subset='train',categories = sci_categories, shuffle = True, random_state = 42)
    sci_test  = fetch_20newsgroups(subset='test',categories = sci_categories, shuffle = True, random_state = 42)

    rec_train = rec_train.data +rec_test.data
    rec_train = rec_train
    sci_train = sci_train.data[:40]
    rec_target = np.ones(len(rec_train))
    sci_target = np.zeros(40)
    sci_test_target = np.zeros(len(sci_test.data))


    #text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True)),('clf', sklearn.svm.LinearSVC())])
    #text_clf.fit(combinded_train, combined_target)
    #preds = text_clf.predict(sci_test.data)
    #print(preds)
    #c = np.mean(preds == sci_test_target )
    #print(c)

    tfidf_transformer = TfidfTransformer(use_idf=True)
    count_vect = CountVectorizer()
    combinded_train = rec_train + sci_train
    combined_counts = count_vect.fit_transform(combinded_train)
    combined_tf = tfidf_transformer.fit_transform(combined_counts)
    combined_target = np.concatenate((rec_target, sci_target), axis=0)
    rec_counts = count_vect.transform(rec_train)
    sci_counts = count_vect.transform(sci_train)
    sci_test_counts = count_vect.transform(sci_test.data)
    rec_train_tf = tfidf_transformer.transform(rec_counts)
    sci_train_tf = tfidf_transformer.transform(sci_counts)
    sci_test_tf = tfidf_transformer.transform(sci_test_counts)
    combined_tf =sparse.vstack((rec_train_tf,sci_train_tf))


    #preds = model_pipe_predict(sci_test_tf,rec_train_tf,rec_target,sci_train_tf,sci_target,50)
    naive_correct = naive_model_return_error(combined_tf,combined_target,sci_test_tf, sci_test_target)
    print(naive_correct)

    model = Tradaboost(learner="logistic_regression")
    model.fit(rec_train_tf, rec_target, sci_train_tf, sci_target,MAX_ITERATION=100)
    preds = model.predict(sci_test_tf)
    print(preds)
    res = return_correct_rate(preds,sci_test_target)
    print(res)


def naive_model_return_error(train, y, test,test_y):
    """implement a comparative method as a naive model"""
    model = sklearn.linear_model.LogisticRegression()
    model.fit(train,y )
    preds = model.predict(test)
    c= 0
    for i in range(len(preds)):
        if preds[i] == test_y[i] :
            c+=1
    res = c/len(test_y)
    return res


def return_correct_rate(preds, target):
    c= 0
    for i in range(len(preds)):
        if preds[i] == target[i] :
            c+=1
    res = c/len(target)
    return res




if __name__ == "__main__":
    main()


