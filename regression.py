import os, cv2
import matplotlib.pyplot as plt
import numpy as np

def same_size(X,y):
    minlength = min(y.shape[0],X.shape[0])
    print minlength
    y = y[:minlength]
    X = X[:minlength]
    return X,y

#data = np.load('cloth2/controller/data.npz')
def load_model_name(data_name):
    data = np.load(data_name)
    pos = data['pos']
    feat = data['feat']
    return load_model_data(pos,feat)
    
    
def load_model_data(pos, feat, num_samples, alpha):
    inds = np.random.choice(a=min(len(feat),len(pos)),size=num_samples*len(feat))
    if len(inds)%2 == 1:
        inds=inds[:-1]
    nd = len(inds)/2 # number of deltas
    X_train = feat[inds[:nd]] - feat[inds[nd:]]
    y_train = pos[inds[:nd]] - pos[inds[nd:]]
    X_train, y_train = same_size(X_train, y_train)
    print X_train.shape, y_train.shape

    # begin training
    from sklearn import linear_model
    model = linear_model.Lasso(alpha = alpha)
    model.fit(X_train, y_train)
    return model

def combine_feature(data_name1, data_name2):
    data1 = np.load(data_name1)
    pos = data1['data']
    feat = data1['feat']
    
    data2 = np.load(data_name2)
    feat2 = data2['feat']
    print feat1.shape,feat2.shape
    print feat1[0]
    
    print feat2[0]
    feat = np.hstack((feat1,feat2))
    return feat, pos



