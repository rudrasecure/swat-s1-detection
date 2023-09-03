import pandas as pd
import numpy as np 

# data = pd.read_csv("test.csv")
# data=data.drop(labels=["Time"],axis=1)
# test = np.asarray(data).astype('float32')/255
from numpy import dot
from numpy.linalg import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

encoder = load_model("model.h5")
data = pd.read_csv("/home/mukesh/minicps/examples/swat-s1/logs/data.csv")
data=data.drop(labels=["Time"],axis=1)
test = np.asarray(data).astype('float32')/255
result = encoder.predict(test)
for i in range(0,len(test)):
    a=result[i]
    b=test[i]
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    if cos_sim<0.64:
        x = test[i].astype("float32")*255
        print(f"[*] ALERT! Anomaly detected! Score:{cos_sim}\nData:{x}")
