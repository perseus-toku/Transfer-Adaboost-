import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.svm
from scipy import sparse


class Tradaboost:

    def __init__(self, learner="logistic_regression", quiet = False):
        """Initialized all meta data--> what model to use as the base learner """
        self.learner = learner
        self.metadata = []
        self.quiet=quiet


    def Learner(self, X, y, W):
        """Supports SVM, LR"""
        learner = self.learner
        if learner == "logistic_regression":
            sum_w = sum(W)
            P = [x / sum_w for x in W]
            model = sklearn.linear_model.LogisticRegression(C=sum_w)
            model.fit(X, y, sample_weight=P)
            preds = model.predict(X)
            c= 0
            for i in range(len(preds)):
                if preds[i] == y[i] :
                    c+=1
            return model

        elif learner == "linear_svm":
            sum_w = sum(W)
            P = [x / sum_w for x in W]
            model = sklearn.svm.LinearSVC(C=sum_w)
            model.fit(X, y, sample_weight=P)
            preds = model.predict(X)
            c= 0
            for i in range(len(preds)):
                if preds[i] == y[i] :
                    c+=1
            return model

        elif learner == "svm":
            sum_w = sum(W)
            P = [x / sum_w for x in W]
            model = sklearn.svm.SVC(C=sum_w)
            model.fit(X, y, sample_weight=P)
            preds = model.predict(X)
            c = 0
            for i in range(len(preds)):
                if preds[i] == y[i]:
                    c += 1
            return model

        else:
            raise ValueError("name not found")


    def fit(self, diff_train, diff_label, same_train, same_train_label, MAX_ITERATION=30):
        """split same set to train set and test set """
        #same distribution split
        n,m = diff_train.shape[0], same_train.shape[0]
        #set up initial weight
        W = [1.0 for i in range(m+n)]
        MODEL_LIST, ERROR_LIST,BETA_T_LIST =[],[],[]
        combined_train = sparse.vstack((diff_train,same_train))
        combined_y = np.concatenate((diff_label, same_train_label), axis=0)
        beta = 1.0 / (1.0 + math.sqrt(2.0 * math.log(float(n + 1) / MAX_ITERATION)))
        beta_t = 0

        for t in range(MAX_ITERATION):
            if not self.quiet:print("----iteration number {} ----".format( t+1 ))
            model = self.Learner(combined_train.A, combined_y, W)
            preds = model.predict(combined_train)
            error = 0.0
            for i in range(n,m+n):
                error += (W[i] * abs(preds[i] - combined_y[i]))
            error  = error / sum(W[n:])

            if error !=1:
                beta_t = error/(1.0-error)
            if error >= 0.5 and error !=1:
                #coonsider more about this
                if not self.quiet:print("error is greater than 1/2 : %f", error)
                beta_t =0.45/(0.51)
            if error == 1:
                beta_t= 0.4

            if error == 0:
                # if error is zero, we still want the learning to continue
                #so continue update the weights see the change
                beta_t = 0.99

            if not self.quiet:
                print("error is: ", error)
                print("weight first 10", W[:10])
                print("weight last 10", W[-10:])
                print("fist 10",preds[:10])
                print("last 10", preds[-10:])

            for i in range(n):
                W[i] = W[i] * (beta ** (abs(preds[i] - combined_y[i])))
            for j in range(n, m + n):
                W[j] = W[j] * (beta_t ** ((-1) * abs( preds[j] - combined_y[j])))

            ERROR_LIST.append(error)
            MODEL_LIST.append(model)
            BETA_T_LIST.append(beta_t)


        r = int( MAX_ITERATION/2)
        self.metadata.append(MODEL_LIST[r:])
        self.metadata.append(ERROR_LIST[r:])
        self.metadata.append(BETA_T_LIST[r:])



    def predict_one(self,x, MODEL, ERROR, BETA_T):
        """x is a single test sample """
        base = 1
        pred = 1
        for i in range(len(MODEL)):
            model = MODEL[i]
            hx= model.predict(x)[0]
            if BETA_T[i] == 0:
                #avoid division by zero
                continue
            pred *= BETA_T[i] ** ((-1) * hx )
            base *= BETA_T[i] ** (-1 / 2)

        if pred >= base:
            return 1
        else:
            return 0


    def model_predict(self,X, MODEL, ERROR, BETA_T):
        preds =[]
        for i in range(X.shape[0]):
            preds.append(self.predict_one(X[i], MODEL, ERROR, BETA_T))
        return preds


    def predict(self, test):
        """test is the test data
            rtype: preds is a list of predictions
        """
        if len(self.metadata) ==0:
            raise ValueError("no fitted model found")
        MODEL_LIST, ERROR_LIST, BETA_T_LIST = self.metadata[0],  self.metadata[1],self.metadata[2]
        preds= self.model_predict(test, MODEL_LIST,ERROR_LIST,BETA_T_LIST)
        return preds


    def calculate_error(self, preds, target):
        c = 0
        for i in range(len(preds)):
            if preds[i] == target[i]:
                c += 1
        res = round( 1.0 - c / len(target), 4 )
        return res







