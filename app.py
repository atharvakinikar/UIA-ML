import numpy as np
from flask import Flask,request,jsonify
import pickle
from flask import Flask, request
import os
from flask_cors import CORS,cross_origin
import pandas as pd
import seaborn as sns
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import boto3
import json
import base64
import tinys3
import os

app = Flask(__name__)
cors=CORS(app,resources={r'/api/*':{'origins':'*'}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['PROPAGATE_EXCEPTIONS'] = True
app.secret_key = 'jose'

textractclient=boto3.client("textract",aws_access_key_id=os.environ.get("aws_access_key_id"),aws_secret_access_key=os.environ.get("aws_secret_access_key"),region_name="us-east-1")

@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,HEAD,OPTIONS,POST,PUT"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept, Authorization"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return ("Hi!")

@app.route('/predict',methods = ['POST'])
def predict():
    val = request.get_json(force=True)
    print(type(val['FVC']))
    AGE=int(val['AGE'])
    FEV1=float(val['FEV1'])
    FEV1PRED=float(val['FEV1PRED'])
    FVC=float(val['FVC'])
    FVCPRED=int(val['FVCPRED'])
    SGRQ=float(val['SGRQ'])
    AGEquartiles=int(val['AGEquartiles'])
    gender=int(val['gender'])
    smoking=int(val['smoking'])
    Diabetes=int(val['Diabetes'])
    muscular=int(val['muscular'])
    hypertension=int(val['hypertension'])
    AtrialFib=int(val['AtrialFib'])
    IHD=int(val['IHD'])
    data = pd.read_csv('dataset.csv')
    data = data.drop(['COPDSEVERITY','PackHistory','ID','MWT1','MWT2','CAT','HAD','MWT1Best','Unnamed: 0'], axis=1)
    data['copd']=data['copd'].factorize()[0]
    X = data.drop('copd',axis=1).values
    y = data['copd']

    # 20% test 80% train
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15,random_state = 40)
    XGClass_model = XGBClassifier(learning_rate=0.01, n_estimators=1500, max_depth=4,num_class=len(y.unique()))
    XGClass_model.fit(X_train,y_train)
    output = XGClass_model.predict([[AGE,FEV1,FEV1PRED,FVC,FVCPRED,SGRQ,AGEquartiles,gender,smoking,Diabetes,muscular,hypertension,AtrialFib,IHD]])[0]
    if(output==0):
        return("1")
    elif(output==1):
        return("2")
    elif(output==2):
        return("3")
    else:
        return("4")

# @app.route('/ocr',methods = ['POST'])
# @cross_origin(origin='*')
# def getValues():
#     file=request.files.get("filename")
#     binaryFile=file.read()
#     response = textractclient.detect_document_text(
#         Document={
#             'Bytes': binaryFile
#         }
#     )
#     ans={}

#     for i in range(0,len(response['Blocks'])):
#         if(response['Blocks'][i]["BlockType"]) == "LINE":
#             if(response['Blocks'][i]["Text"]=="pH"):
#                 ans['pH']=response['Blocks'][i+1]["Text"]
#                 # extractedText=" pH "+extractedText+response['Blocks'][i+1]["Text"]+" "
#             if(response['Blocks'][i]["Text"]=="pCO2"):
#                 x=response['Blocks'][i+1]["Text"]
#                 x=re.sub("[^\d\.]", "", x)
#                 ans['pCO2']=x
#             if("Na" in response['Blocks'][i]["Text"] or "Sodium" in response['Blocks'][i]["Text"]):
#                 x=response['Blocks'][i+1]["Text"]
#                 x=re.sub("[^\d\.]", "", x)
#                 ans['Na']=x
#             if("cHCO" in response['Blocks'][i]["Text"] or "HCO3" in response['Blocks'][i]["Text"]):
#                 x=response['Blocks'][i+1]["Text"]
#                 x=re.sub("[^\d\.]", "", x)
#                 ans['HCO3']=x
#             if("cK" in response['Blocks'][i]["Text"] or "Potassium" in response['Blocks'][i]["Text"]):
#                 x=response['Blocks'][i+1]["Text"]
#                 x=re.sub("[^\d\.]", "", x)
#                 ans['K']=x
#             if("cCl" in response['Blocks'][i]["Text"] or "cCI" in response['Blocks'][i]["Text"] or "Chloride" in response['Blocks'][i]["Text"] or "cCh" in response['Blocks'][i]["Text"]):
#                 x=response['Blocks'][i+1]["Text"]
#                 x=re.sub("[^\d\.]", "", x)
#                 ans['Cl']=x

#     jsonData=json.dumps(ans)
#     return(jsonData)


@app.route('/ocr',methods = ['POST'])
@cross_origin(origin='*')
def getValues():
    file=request.files.get("myfile")
    binaryFile=file.read()

    # file = request.get_json(force=True)
    # image = file['data'][23:]

    # decoded_data=base64.b64decode((image))
    # report_image=open('report.jpeg', 'wb')
    # report_image.write(decoded_data)
    # report_image.close()

    # conn = tinys3.Connection("AKIA2NCHYUZJ2NZMEVWM","V0EZbrJEtF5oKlRIKV1hMM4xO4X//Xs65wVTMAt8",tls=True)
    # f = open('report.jpeg','rb')
    # conn.upload('report.jpeg',f,'uia-antons')

    # f1=open('report.jpeg','rb')
    # bytes=f1.read()

    # encoded_b2 = "".join([format(n, '08b') for n in image])

    # print(file['data'])

    # session = boto3.Session(
    # aws_access_key_id='AKIA2NCHYUZJ2NZMEVWM',
    # aws_secret_access_key='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',
    # )
    # s3 = session.resource('s3')
    # s3.meta.client.upload_file(Filename='report.jpeg', Bucket='uia-antons')

    response = textractclient.detect_document_text(
        Document={
            'Bytes': binaryFile,
            # "S3Object": {
            #     "Bucket": "uia-antons",
            #     "Name": "report.jpeg",
            # }
        }
    )

    ans={}

    for i in range(0,len(response['Blocks'])):
        if(response['Blocks'][i]["BlockType"]) == "LINE":
            if(response['Blocks'][i]["Text"]=="pH"):
                ans['pH']=response['Blocks'][i+1]["Text"]
                # extractedText=" pH "+extractedText+response['Blocks'][i+1]["Text"]+" "
            if(response['Blocks'][i]["Text"]=="pCO2"):
                x=response['Blocks'][i+1]["Text"]
                x=re.sub("[^\d\.]", "", x)
                ans['pCO2']=x
            if("Na" in response['Blocks'][i]["Text"] or "Sodium" in response['Blocks'][i]["Text"]):
                x=response['Blocks'][i+1]["Text"]
                x=re.sub("[^\d\.]", "", x)
                ans['Na']=x
            if("cHCO" in response['Blocks'][i]["Text"] or "HCO3" in response['Blocks'][i]["Text"]):
                x=response['Blocks'][i+1]["Text"]
                x=re.sub("[^\d\.]", "", x)
                ans['HCO3']=x
            if("cK" in response['Blocks'][i]["Text"] or "Potassium" in response['Blocks'][i]["Text"]):
                x=response['Blocks'][i+1]["Text"]
                x=re.sub("[^\d\.]", "", x)
                ans['K']=x
            if("cCl" in response['Blocks'][i]["Text"] or "cCI" in response['Blocks'][i]["Text"] or "Chloride" in response['Blocks'][i]["Text"] or "cCh" in response['Blocks'][i]["Text"]):
                x=response['Blocks'][i+1]["Text"]
                x=re.sub("[^\d\.]", "", x)
                ans['Cl']=x

    jsonData=json.dumps(ans)
    return(jsonData)


if __name__=="__main__":
    app.run(debug=True)