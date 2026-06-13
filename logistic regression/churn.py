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
model=LogisticRegression()
model=model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#------------------------ evalution---------------------------------------
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test,y_pred))
#---------visuliation--------------------------------
def visuliation(cm):
  plt.figure(figsize=(10,6))
  plt.bar(["true pos","false nagative","false pas","true nagative"],[cm[0,0],cm[0,1],cm[1,0],cm[1,1]],color=['green','red','yellow','blue'])
  plt.title("confusion matrix")
  plt.bar_label(plt.gca().containers[0])
  plt.grid(True)
  plt.show()
visuliation(cm)
