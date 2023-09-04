import select
import subprocess
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import time

# convert list to a Dataframe
def convertToDF(values):
    kv_pair = {"Time": float(values[0]), "MV101": int(values[1]), "P101": float(values[2]), "LIT101": float(
        values[3]), "LIT301": float(values[4]), "FIT101": float(values[5]), "FIT201": float(values[6])}
    dataFrame = pd.DataFrame([kv_pair])
    return dataFrame


# tail last line
def readLastLine(filename):
    line = subprocess.check_output(['tail', '-1', filename])
    return line.decode("utf-8")

def alertGenerator(data):
    print(f"[*] ANOMALY DETECTED. Alert Data:\n{data}\n")

class Detector:

    def __init__(self):
        self.autoencoder = load_model("model.h5")
        self.threshold = 0.64 # threshold can be adjusted based on required sensitivity level

    def detector(self, data):
        data=data.drop(labels=["Time"],axis=1)
        test = np.asarray(data).astype('float32')/255
        result = self.autoencoder.predict(test)
        # calculate cosine similarity. Value > 0.85 is desirable
        a=result[0]
        b=test[0]
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        return cos_sim
    
    # return True to trigger alarm, else do nothing
    def alarm(self, cos_sim):
        if cos_sim > self.threshold:
            return False
        else:
            return True

# initialize the detector outside the loop
detect = Detector()
f = subprocess.Popen(['tail','-n', '+1', '-F', '../examples/swat-s1/logs/data.csv'], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
p = select.poll()
p.register(f.stdout)


while True:
    if p.poll(1):
        line = f.stdout.readline().decode('utf-8')
        lineMod = line.split(",")
        data = convertToDF(lineMod)
        if detect.alarm(detect.detector(data)):
            alertGenerator(data)
