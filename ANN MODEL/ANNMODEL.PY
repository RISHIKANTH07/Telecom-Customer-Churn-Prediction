from pandas.plotting import table
from imblearn import over_sampling
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
data_train=pd.read_csv('/content/customer_churn_dataset-training-master.csv')
df=pd.DataFrame(data_train)
df=df.dropna()



df["Subscription Type"] = df["Subscription Type"].replace({
    "Basic": 0,
    "Premium": 1,
    "Standard": 2
})

df["Contract Length"] = df["Contract Length"].replace({
    "Annual": 1,
    "Quarterly": 2,
    "Monthly": 3
})
df['Gender']=(df['Gender']=='Male').astype(int)
df['Gender']=(df['Gender']=='Female').astype(int)
print(df)
x=df.drop("Churn",axis=1)
y=df["Churn"]
#---------------------------train  data------------------------------
data_test=pd.read_csv('/content/customer_churn_dataset-testing-master.csv')
df_test=pd.DataFrame(data_test)
df_test=df_test.dropna()
df_test=df_test.drop(columns=['CustomerID'])
x1=df.drop("Churn",axis=1)
y1=df["Churn"]


def scaling(x,y,oversample):
  scale=StandardScaler()
  x=scale.fit_transform(x)
  if(oversample):
    rov=RandomOverSampler()
    rov.fit(x,y)
  return x,y
#_____________split and train___________________________________________
x_train,y_train=scaling(x,y,oversample=True)
x_test,y_test=scaling(x1,y1,oversample=False)
#-----------------------------------------------------------------------
import tensorflow as tf

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(45000, activation='relu', input_shape=(440832,)),
    tf.keras.layers.Dense(45000, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
history =nn_model.fit(x_train,y_train,epochs=20,batch_size=64,validation_split=0.2,verbose=1)
y_pred = nn_model.predict(x_test)
print(y_pred)
