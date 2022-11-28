from datetime import datetime, timedelta
import os
from datetime import date
import json
import time
import sys
from airflow import DAG
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from airflow.operators.python import PythonOperator
import numpy as np
import ssl
import sys
from smtplib import SMTP_SSL as SMTP      # this invokes the secure SMTP protocol (port 465, uses SSL)
from email.mime.text import MIMEText
from sklearn.preprocessing import MinMaxScaler
import boto3

ssl._create_default_https_context = ssl._create_unverified_context
stock_code = "HPG"

def craw_stock_price(**kwargs):

    to_date = kwargs["to_date"]
    from_date = "2000-01-01"

    stock_price_df = pd.DataFrame()
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    url = "https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{}~date:gte:{}~date:lte:{}&size=9990&page=1".format(stock_code, from_date, to_date)
    print(url)


    from urllib.request import Request, urlopen

    req = Request(url, headers={'User-Agent': 'Mozilla / 5.0 (Windows NT 6.1; WOW64; rv: 12.0) Gecko / 20100101 Firefox / 12.0'})
    x = urlopen(req, timeout=10).read()

    req.add_header("Authorization", "Basic %s" % "ABCZYXX")

    json_x = json.loads(x)['data']
    with open("data_{}.json".format(stock_code), "w") as outfile:
      json.dump(json_x, outfile)

    return True

def extract_from_json(file_to_process):
  columns =['code', 'date', 'basicPrice']
  stock_df = pd.DataFrame()
  with open(file_to_process, 'r') as f:
    json_x = json.load(f)
  for stock in json_x:
    stock_df = stock_df.append(stock, ignore_index=True)
  return stock_df[columns]

def transform(data):
  data.loc[:,'basicPrice'] = round(data.basicPrice, 1)
  return data

def load(targetfile,data_to_load):
    data_to_load.to_csv(targetfile, index=None)

csv_filename = "stock_{}.csv".format(stock_code)
bucket_name = 'data-stock-bucket'

def etl(): 
    data_stock = extract_from_json('data_{}.json'.format(stock_code))  
    data_stock = transform(data_stock)
    load(csv_filename, data_stock)
    return True 

def upload_s3():
    AWS_KEY_ID=""  
    AWS_SECRET=""
    bucket_name = 'data-stock-bucket'
    # Generate the boto3 client for interacting with S3
    s3 = boto3.client('s3', region_name='ap-southeast-1', 
                            # Set up AWS credentials 
                            aws_access_key_id=AWS_KEY_ID, 
                            aws_secret_access_key=AWS_SECRET)
    s3.upload_file(
    # Complete the filename
    Filename=csv_filename, 
    # Set the key and bucket
    Key=csv_filename, 
    Bucket=bucket_name,
    # During upload, set ACL to public-read
    ExtraArgs = {
        'ACL': 'public-read'})
    return True

def train_model():
    # Doc du lieu 
    dataset_train = pd.read_csv('stock_HPG.csv')
    training_set = dataset_train.iloc[:, 2:3].values
    training_set_revese = training_set[::-1]

    # Thuc hien scale du lieu gia ve khoang 0,1
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set_revese)

    # Tao du lieu train, X = 60 time steps, Y =  1 time step
    X_train = []
    y_train = []
    no_of_sample = len(training_set)

    for i in range(60, no_of_sample):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Xay dung model LSTM
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=10, batch_size=32)
    regressor.save("stockmodel_{}.h5".format(stock_code))
    return True

def stock_predict():
    #load model
    model = load_model("stockmodel_{}.h5".format(stock_code))

    # Doc du lieu 
    dataset_train = pd.read_csv('stock_{}.csv'.format(stock_code))
    training_set = dataset_train.iloc[:, 2:3].values
    training_set_revese = training_set[::-1]

    # Thuc hien scale du lieu gia ve khoang 0,1
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set_revese)

    x_pre = training_set_scaled[-60:].reshape(1,-1)
    x_pre = np.array(x_pre)
    x_pre = np.reshape(x_pre, (x_pre.shape[0], x_pre.shape[1], 1))
    print(x_pre.shape)
    ynew = model.predict(x_pre)
    y_real = sc.inverse_transform(ynew)

    # save predict result to file 
    f = open("result.txt", "w")
    f.write(str(y_real[0][0]))
    f.close()

    return True

def email(): 
    # send email use SMTP 
    SMTP_Host = 'smtp.gmail.com'
    sender = ''
    receivers = ['']
    username = ""
    password = ""

    #open and read the file
    f = open("result.txt", "r")
    predict_result = f.read()
    f.close()

    # typical values for text_subtype are plain, html, xml
    import datetime
    import pytz
    now = datetime.datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
    dt_string = now.strftime("%d/%m/%Y")

    text_subtype = 'plain'
    content = "Kết quả dự đoán cổ phiếu {} ngày {} là {}".format(stock_code, dt_string, predict_result)
    subject = "KẾT QUẢ DỰ ĐOÁN CỔ PHIẾU"
    try:
        msg = MIMEText(content, text_subtype)
        msg['Subject'] = subject
        msg['From'] = sender  # some SMTP servers will do this automatically, not all
        conn = SMTP(SMTP_Host)
        conn.set_debuglevel(False)
        conn.login(username, password)
        try:
            conn.sendmail(sender, receivers, msg.as_string())
        finally:
            conn.quit()
    except Exception as error:
        print(error)
    
    return True

dag = DAG(
    'ML_pipeline',
    default_args={
        'email': ['pvdphong.98@gmail.com'],
        'email_on_failure': True,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='A ML training pipline DAG',
    schedule_interval="0 0 * * 1-5",
    start_date= datetime.today() - timedelta(days=1),
    tags=['phongpham'])


crawl_data = PythonOperator(
    task_id='crawl_data',
    python_callable=craw_stock_price,
    op_kwargs={"to_date": "{{ ds }}"},
    dag=dag
)

etl_data = PythonOperator(
    task_id='etl_data',
    python_callable=etl,
    dag=dag
)

load_s3 = PythonOperator(
    task_id='load_s3',
    python_callable=upload_s3,
    dag=dag
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

stock_predict = PythonOperator(
    task_id='stock_predict',
    python_callable=stock_predict,
    dag=dag
)

email_operator = PythonOperator(
    task_id='email_operator',
    python_callable=email,
    dag=dag
)

crawl_data >> etl_data >> train_model >> stock_predict >> email_operator
etl_data >> load_s3