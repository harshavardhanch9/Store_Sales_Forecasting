import streamlit as st
import data_handler as dh
from model import MLP
from PIL import Image
import pandas as pd
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
import torch
from torch import nn
# import torch.nn.functional as F
#import seaborn as sns
import random
from PIL import Image 

logo = Image.open("logo.jpg")
st.image(logo,use_column_width=True)

# Text/Title
st.header("Store Sales Prediction of Corporación Favorita")



st.sidebar.info("Start Sales Forecasting" )
st.sidebar.write("## Task")

task = st.sidebar.selectbox("Choose the task?",("Exploratory Data Analysis", "Model prediction"))

st.sidebar.write('Selected:', task)

if task == "Exploratory Data Analysis":

        file = st.sidebar.file_uploader("Upload your input train.csv file", type=["csv"])

        st.markdown(""" Here we can see some analysis""")

        #data = pd.read_csv(file)
        #data1 = pd.read_csv("base_train.csv")

        # st.header("First we see the raw data")
        # #if st.button("Show the raw data"):
        #st.dataframe(data= data1[0:20])

        st.subheader("The sales of all stores located in ")
        img = Image.open("sales.png")
        if st.button("Show the sales over the years"):
                st.image(img, caption="Store sales for all the days", use_column_width=True)

        st.subheader("Statewide sales")
        img = Image.open("state.png")
        if st.button("Show the statewide sales"):
                st.image(img, caption="Statewide sales", use_column_width=True)

        st.subheader("The storewide sales")
        img = Image.open("store sales.png")
        if st.button("Show the storewide sales"):
                st.image(img, caption="Storewide sales", use_column_width=True)

        st.subheader("Year and month wise sales")
        img = Image.open("year_month.png")
        if st.button("Show the year wise sales"):
                st.image(img, caption="Year and month wise sales", use_column_width=True)

        st.subheader("The oil prices over the months and years")
        img = Image.open("oil.png")
        if st.button("Show the oil prices"):
                st.image(img, caption="oil prices over the months and years", use_column_width=True)

        
        # st.header("Finally we see the clean data")
        # if st.button("Show the clean data"):
        #         st.dataframe(data= data[0:20])

elif task == "Model prediction":
        file = st.sidebar.file_uploader("Upload your input train.csv file", type=["csv"])

        if file is not None:
                # data = pd.read_csv(file)
                # x = data.drop(["date"], axis=1).values
                # x = to_batches(x, batch_size=100, shuffle = True)
                # x_test = torch.tensor(x.astype(np.float32))
                x_train, x_test, y_train,  y_test =  dh.load_data(file, batch_size=100, shuffle = True)
                
                model = MLP(13,[10,8], 1)

                state_dict = torch.load('store_sales.pth')

                model.load_state_dict(state_dict)

                prediction = model(x_test)



                #st.write(prediction.shape)
                number = st.number_input('Select the batch number to predict the sales', int(0), int(6002))

                pred = prediction.detach().numpy().tolist()[int(number)]
                
                y_pred = y_test.detach().numpy().tolist()[int(number)]

                st.markdown("""The below charts shows the sales prediction for the batch of 100 data""")
                fig, ax = plt.subplots(1, 2,figsize=(11,4))
                ax[0].set_title('Sales Prediction')
                ax[0].plot(pred)
                ax[1].set_title('Actual Sales')
                ax[1].plot(y_pred)
                plt.xlabel('Batch')
                st.pyplot(fig)

