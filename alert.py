import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize']=14,6

#Read CSV File geographical data
accident=pd.read_csv('BookFinal.csv')
accident.head()


#construct model
from sklearn.cluster import KMeans
x=accident.iloc[:,:2]
km=KMeans(76)
km.fit(x)
identified_cluster=km.fit_predict(x)
data_with_cluster=accident.copy()
data_with_cluster['Cluster']=identified_cluster
data_with_cluster.to_csv('file1.csv')

import pickle

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(km, open('model.pkl','wb'))
