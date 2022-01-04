# Tamir Zolzaya
# 08470082

# import library
from scipy.stats.morestats import circmean
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# load dataset
df=pd.read_csv("./winequality-white.csv", sep=";")
df.head(12)
# let's create an APP title
st.title("White Whine Classification")
# let's create an App checkbox
if st.checkbox(' Show Dataframe'):
    st.write(df)

# Show heatmap dataframe 
res_myheatmap69= st. checkbox ("Show Heatmap")
if res_myheatmap69:
    st.subheader ("Heatmap of white wine")
    fig_ht, ax = plt.subplots()
    sns.heatmap(df.corr(), color = "honeydew", ax=ax)
    st.write(fig_ht)

# Create a Bar chart
column1 = df.columns[11:]
column2 = st.selectbox("Choose attributes", df.columns[:11])

# Choose chart and color
st.subheader ("Bar chart")
fig=px.bar (df, x='quality', y=column2)
fig.update_traces(marker_color='rgb(255, 0, 255)', marker_line_color='rgb(255, 0, 128)',marker_line_width=1.5)
st.plotly_chart(fig)

# Train the Model
# Data Preparation
X = df.iloc[:, :-1]. values
y=df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)

# split data to train 70% and 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=2)

# Create a KNN classifier object with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the knn model with fit() function
knn.fit(X_train, y_train)

# make prediction
y_pred = knn.predict (X_test)

# Train the Model RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators=50)
RandomForest.fit(X_train, y_train)
y_pred_RandomForest = RandomForest.predict(X_test)


# Train the Model Decision Tree
dtc = DecisionTreeClassifier()
dtc = dtc.fit(X_train,y_train)
y_pred_dtc = dtc.predict(X_test)

# Choose Model to see Accuracy score
models= st.radio("Choose Model",('KNN Classifier','Random Forest Classification','Decision Tree Classifier'))


# Get the accuracy score and confusion matrix of trained models
if models=='KNN Classifier':
    st.header("Train KNN Classification Model")
    st.write("The Accuracy Score of the Model is:")
    st.success(metrics.accuracy_score(y_test, y_pred))
    st.write("The Confusion matrix of the Model is:")
    st.write(confusion_matrix(y_test, y_pred))
elif models=='Random Forest Classification':
    st.header("Train Random Forest Classification Model")
    st.write("The Accuracy Score of the Model is:")
    st.success(metrics.accuracy_score(y_test, y_pred_RandomForest))
    st.write("The Confusion matrix of the Model is:")
    st.write(confusion_matrix(y_test, y_pred_RandomForest))
elif models=='Decision Tree Classifier':
    st.header("TrainDecision Tree Classifier Model")
    st.write("The Accuracy Score of the Model is:")
    st.success(metrics.accuracy_score(y_test, y_pred_dtc))
    st.write("The Confusion matrix of the Model is:")
    st.write(confusion_matrix(y_test, y_pred_dtc))

# Prediction of Wine Quality
st.sidebar.header ("White Wine Quality Prediction")

# Display the input of attibutes
fixed_acidity=np.float(st.sidebar.slider("Fixed Acidity ",min_value=3.8,max_value=14.2,step=0.1))
volatile_acidity=np.float(st.sidebar.slider("Volatile Acidity",min_value=0.08,max_value=1.1,step=0.01))
citric_acid=np.float(st.sidebar.slider("Citric acid",min_value=0.0,max_value=1.66,step=0.01))
residual_sugar=np.float(st.sidebar.slider("Residual sugar",min_value=0.6,max_value=65.8,step=0.1))
chlorides=np.double(st.sidebar.slider("Chlorides",min_value=0.009,max_value=0.34,step=0.001))
free_sulfur_dioxide=np.float(st.sidebar.slider("Free sulfur dioxide",min_value=2.0,max_value=289.0,step=2.0))
total_sulfur_dioxide=np.float(st.sidebar.slider("Total sulfur dioxide",min_value=9.0,max_value=440.0,step=2.0))
density=np.double(st.sidebar.slider("Density",min_value=0.98,max_value=1.03,step=0.01))
pH=np.float(st.sidebar.slider("pH",min_value=2.72,max_value=3.82,step=0.01))
sulphates=np.float(st.sidebar.slider("Sulphates",min_value=0.22,max_value=1.08,step=0.01))
alcohol=np.float (st.sidebar.slider("Alcohol",min_value=8.0,max_value=14.2,step=0.1))
predict_button=st.sidebar.button("Predict ")

# Confirm Prediction of Decision Tree Classification
if predict_button:
    data=(fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)
    data_arr= np.asarray(data).reshape(1,-1)
    prediction=dtc.predict(data_arr)
    if prediction==0:
        predict_res= "Bad Quality"
    else:
        predict_res="Good Quality"
    st.subheader("Quality Forecast")
    st.success("The predict result is "+predict_res)