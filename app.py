from flask import Flask, render_template, request
import io
from io import StringIO
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

def test_feature_engineering(df):
    df.columns = ['age','workclass','fnlwgt','education','ed_num','marital_status','occupation','relation',
              'race','sex','cap_gain','cap_loss','hrs_per_week','country']
    
    df['workclass'] = np.where(df['workclass'] == ' ?', 'Missing', df['workclass'])
    wrkcls_label = {value:key for key,value in enumerate(df['workclass'].unique())}
    df['workclass'] = df['workclass'].map(wrkcls_label)
    
    edn_label = {value:key for key,value in enumerate(df['education'].unique())}
    df['education'] = df['education'].map(edn_label)
    
    mrg_label = {value:key for key,value in enumerate(df['marital_status'].unique())}
    df['marital_status'] = df['marital_status'].map(mrg_label)

    df['occupation'] = np.where(df['occupation'] == ' ?', 'Missing', df['occupation'])
    occp_label = {value:key for key,value in enumerate(df['occupation'].unique())}
    df['occupation'] = df['occupation'].map(occp_label)
    
    rel_label = {value:key for key,value in enumerate(df['relation'].unique())}
    df['relation'] = df['relation'].map(rel_label)
    
    race_label = {value:key for key,value in enumerate(df['race'].unique())}
    df['race'] = df['race'].map(race_label)
    
    df['country'] = np.where(df['country'] == ' ?', 'Missing', df['country'])
    cntry_label = {value:key for key,value in enumerate(df['country'].unique())}
    df['country'] = df['country'].map(cntry_label)
    
    df['sex'] = np.where(df['sex'] == ' Male', 0, 1)
    
    return df

def scalar(df):
    sc = StandardScaler()
    X = df[['age','workclass','fnlwgt','education','ed_num','marital_status','occupation','relation',
              'race','sex','cap_gain','cap_loss','hrs_per_week','country']]
    X = sc.fit_transform(X)
    return X

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/calculate',methods=['POST'])
def predict():
    f = request.files['data_file']
    if not f:
        return render_template('index.html',prediction_text='No File Selected, Try again')
    
    stream = StringIO(f.stream.read().decode('UTF-8'))
    result = stream.read()
 
    df = pd.read_csv(StringIO(result))

    df = test_feature_engineering(df)

    X = scalar(df)

    loaded_model = pickle.load(open('test.pkl','rb'))

    result = loaded_model.predict(X)

    return render_template('index.html',prediction_text = 'Predicted Salaries were {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)