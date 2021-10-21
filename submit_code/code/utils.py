# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import math
import gc


from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


from .fetch_data import train_data


def create_features(data):
    data["time"] = pd.to_datetime(data["time"])
    data["hour"] = data["time"].dt.hour*60+data["time"].dt.minute

    for num in ["hour"]:
        st = preprocessing.StandardScaler()
        data[num] = st.fit_transform(data[[num]].values)

    for fe in ['account', 'IP', 'url','switchIP']:
        le = preprocessing.LabelEncoder()
        data[fe] = le.fit_transform(data[fe])

    return data


cate_features = ['account', 'IP', 'url', 'switchIP']
num_features = ["hour"]


def get_model():
    cat_input = []
    cat_emb = []
    for cat in cate_features:
        cat_inp = Input(shape=(1,), name=cat)
        cat_input.append(cat_inp)
        embed = Embedding(data[cat].max()+1, 200, input_length=1, trainable=True)(cat_inp)
        cat_emb.append(Flatten()(embed))

    num_input=[]
    for num in num_features:
        num_inp = Input(shape=(1,), name=num)
        num_input.append(num_inp)

    x = concatenate(cat_emb+num_input, axis=1)
    fc1 = Dense(1024, activation='relu')(x)
    fc2 = Dense(512, activation='relu')(fc1)
    fc2 = Dense(256, activation='relu')(fc2)
    output = Dense(1, activation="relu")(fc2)
    model = Model(inputs=cat_input+num_input, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    model.summary()
    return model


from sklearn.model_selection import KFold
folds = 5
seed = 2018
skf = KFold(n_splits=folds,shuffle=True, random_state=seed)

te_pred = np.zeros((train_data.shape[0], 1))
test_pred = np.zeros((test_data.shape[0], 1))
test_pred_cv = np.zeros((folds, test_data.shape[0], 1))

cnt = 0
score= 0
score_cv_list=[]
for ii,(idx_train, idx_val) in enumerate(skf.split(train_data)):
    X_train_tr={}
    X_train_te={}
    for col in cate_features+num_features:
        X_train_tr[col] = train_x[col][idx_train]
        X_train_te[col] = train_x[col][idx_val]


    y_tr=train_y[idx_train]
    y_te=train_y[idx_val]

    model = get_model()
    early_stop = EarlyStopping(patience=2)
    check_point = ModelCheckpoint(path+'best_model.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    plateau = ReduceLROnPlateau(
        monitor='val_loss', factor=0.75, patience=5, verbose=0,
        mode='min')

    history = model.fit(X_train_tr, y_tr, batch_size=64, epochs=100, verbose=1, validation_data=(X_train_te,y_te),
                        callbacks=[check_point,plateau])

    model.load_weights(path+'best_model.hdf5')

    preds_te = model.predict(X_train_te,batch_size=1024)
    score_cv = (1/(1+np.sin(np.arctan(mean_squared_error(y_te, preds_te)**0.5))))
    score_cv_list.append(score_cv)
    print(score_cv_list)
    te_pred[idx_val] = preds_te
    preds = model.predict(test_x,batch_size=1024)
    test_pred_cv[ii, :] = preds
    #break

test_pred[:]=test_pred_cv.mean(axis=0)

score=(1/(1+np.sin(np.arctan(mean_squared_error(train_y, te_pred)**0.5))))
score=str(score)[:7]
print(score)

# sub=test[["id"]].copy()
# sub["ret"]=test_pred
# sub.to_csv(path+"sub_nn_%s.csv"%m,index=None)