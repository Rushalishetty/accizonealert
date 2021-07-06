import numpy as np
import pandas as pd
from flask import request
from flask import Flask, render_template,request
import pickle#Initialize the flask App
from math import radians, cos, sin, asin, sqrt
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    lon1=final_features[0][0]
    lat1=final_features[0][1]
    prediction = model.predict([np.array(int_features)])
    print("prediction")
    print(prediction)
    output = round(prediction[0], 2)
    print("output")
    print(output)
    accident=pd.read_csv('file1.csv')
    df_new = accident[accident['Cluster'] == output]
    latt2=df_new['Latitude']
    long2=df_new['Longitude']
    print("current location")
    print("longitude = ",lon1)
    print("lotitude = ",lat1)

    print("predicted location")
    l1=long2.values.tolist()
    lon2=l1[0]
    l2=latt2.values.tolist()
    lat2=l2[0]
    print("longitude = ",lon2)
    print("latitude = ",lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    finaloutput=c * r
    m=finaloutput*1000

    z=((180000*m)/302430)
    print("distance")
    print(z," meter")
    return render_template('after.html', data=m)

if __name__ == "__main__":
    app.run(debug=True)
