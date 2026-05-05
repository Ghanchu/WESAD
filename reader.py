import pandas as pd
import math
import numpy as np
import pickle 

## read in the picke file 

f = open("Data/S17/S17.pkl", "rb")
data = pickle.load(f, encoding='latin1')
print(data)
f.close()