#!/usr/bin/env python
# coding: utf-8

# ## 1.Objective

# - To predict the future stock price of IDEA.NS for next 30 days.

# ## What is LSTM ?
# 
# 
# 
# - Long short-term memory is an artificial recurrent neural network architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can process not only single data points, but also entire sequences of data.
# 
# - LSTM is a type of brain-like system called a neural network. It’s special because it can remember things for a long time. This is really useful when you’re dealing with things that change over time, like weather patterns, stock prices, or language.
# 
# - Here’s how it works:
# 
# - Input Gate: Decides what new information we’re going to store in memory.
# - Forget Gate: Decides what information we’re going to throw away.
# - Output Gate: Decides what information we’re going to use to make decisions.
# - So, LSTM looks at new information (input), decides what to remember and what to forget, and then uses that memory to make decisions.

#  ## 3.Importing Library

# In[1]:



import pandas as pd
import numpy as np
import math
import datetime as dt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[2]:


df=pd.read_csv(r"C:\Users\Armaan\Downloads\IDEA.NS.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# ## 4. Exploratory data analysis.

# In[5]:


df.describe()


# In[6]:


print(df.columns)


# In[7]:


df.isnull().sum()


# ### Insight
# 
# - In the data their are 6 Nan values.

# In[8]:


df=df.dropna()


# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# ### 4.1 Staring Date and End Date.

# In[11]:


sd=df.iloc[0][0]
ed=df.iloc[-1][0]

print('Starting Date',sd)
print('Ending Date',ed)


# ### 4.2 Graphical representation

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


fig, axs = plt.subplots(2, 2, figsize=(18, 11))

# Create KDE plots with labels
sns.kdeplot(df.Open, shade=True, color='Green', alpha=0.5, ax=axs[0, 0], label='Open')
sns.kdeplot(df.High, shade=True, color='orange', alpha=0.5, ax=axs[0, 1], label='High')
sns.kdeplot(df.Low, shade=True, color='violet', alpha=0.5, ax=axs[1, 0], label='Low')
sns.kdeplot(df.Close, shade=True, color='blue', alpha=0.5, ax=axs[1, 1], label='Close')

# Add legends
axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

plt.show()


# In[14]:


import plotly.express as px
from itertools import cycle


names= cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])


fig = px.line(df, x='Date', y=["Open",'Close',"High","Low"], labels={'Date': 'Date','value':'Stock value'},
              title='IDEA.NS stock prices chart')

fig.update_layout( font_color='black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))



fig.show()


# In[15]:



fig = px.line(df, x='Date', y='Volume', labels={'Date': 'Date', 'Volume': 'Stock Volume'},
              title='Stock Volume Over Time')

fig.update_layout(font_size=13,font_color='black')

fig.show()


# ### 4.3 Moving Average

# In[16]:


ma50=df.Close.rolling(50).mean()
ma100=df.Close.rolling(100).mean()

ma100


# In[17]:


plt.figure(figsize=(22,8))
plt.plot(df.Open, label='Open')
plt.plot(ma50, "yellow", label='MA50')
plt.plot(ma100, "r", label='MA100')
plt.title('Stock Close Over Time')
plt.grid(True)
plt.legend(loc='best', prop={'size': 16})
plt.show()




plt.figure(figsize=(22,8))
plt.plot(df.Close, label='Close')
plt.plot(ma50, "yellow", label='MA50')
plt.plot(ma100, "r", label='MA100')
plt.title('Stock Open Over Time')
plt.grid(True)
plt.legend(loc='best', prop={'size': 16})
plt.show()


# In[18]:


plt.figure(figsize=(22,8))
plt.plot(df.High, label='High')
plt.plot(ma50, "yellow", label='MA50')
plt.plot(ma100, "r", label='MA100')
plt.title('Stock Close Over Time')
plt.grid(True)
plt.legend(loc='best', prop={'size': 16})
plt.show()



plt.figure(figsize=(22,8))
plt.plot(df.Low, label='low')
plt.plot(ma50, "yellow", label='MA50')
plt.plot(ma100, "r", label='MA100')
plt.title('Stock Open Over Time')
plt.grid(True)
plt.legend(loc='best', prop={'size': 16})
plt.show()


# #### Insight
# 
# - Moving Average of 50 and 100 were plotted along with the original line plot but i have taken MA100 for further analysis because of its smoothness.
# - Moving averages in stock price analysis help identify trends, smooth out price volatility, and indicate support and resistance levels.

# ## 5. MinMaxScaler

# In[19]:


df1=df.Open


# In[20]:


from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler(feature_range=(0,1))
df1=Scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[21]:


df.shape


# In[22]:


df1


# ### 5.1 Train test model

# In[23]:


training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[24]:


print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)


# In[25]:


def create_dataset(dataset, time_step=1):
    dataX = [dataset[i:(i+time_step), 0] for i in range(len(dataset)-time_step-1)]
    dataY = [dataset[i + time_step, 0] for i in range(len(dataset)-time_step-1)]
    return np.array(dataX), np.array(dataY)


# In[26]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)


# In[27]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)


# In[28]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[29]:


tf.__version__


# In[30]:


tf.keras.backend.clear_session()
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# - tf.keras.backend.clear_session(): This clears any old models or layers that might be in memory.
# 
# - model=Sequential(): This starts building a new model.
# 
# - model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1))): This adds a layer to the model that can remember patterns in a sequence of data. It’s set up to look at time_step number of data points at a time.
# 
# - model.add(LSTM(50,return_sequences=True)): This adds another layer that can remember patterns. This layer takes the output of the previous layer as its input.
# 
# - model.add(LSTM(50)): This adds a third layer that can remember patterns. This layer only returns the final output, not the whole sequence.
# 
# - model.add(Dense(1)): This adds a layer that connects everything together. It takes the output of the previous layer and transforms it into the final prediction.
# 
# - model.compile(loss='mean_squared_error',optimizer='adam'): This sets up the model for training. It will use the ‘mean_squared_error’ method to measure how well the model is doing, and the ‘adam’ method to improve the model.
# 
# 

# In[31]:


model.summary()


# In[32]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[33]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[34]:


train_predict.shape, test_predict.shape


# ### 5.2 Retransformation

# In[35]:


# Transform back to original form

train_predict = Scaler.inverse_transform(train_predict)
test_predict = Scaler.inverse_transform(test_predict)


# In[36]:


original_ytrain = Scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = Scaler.inverse_transform(y_test.reshape(-1,1)) 


# In[37]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score


# In[38]:


math.sqrt(mean_squared_error(y_train,train_predict))


# In[39]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[40]:


# Evaluation metrices RMSE and MAE
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Traint data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))


# In[41]:


# shift train predictions for plotting

look_back=time_step
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original Open price','Train predicted Open price','Test predicted Open price'])


plotdf = pd.DataFrame({'Date': df['Date'],
                       'original_Open': df['Open'],
                      'train_predicted_Open': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_Open': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['Date'], y=[plotdf['original_Open'],plotdf['train_predicted_Open'],
                                          plotdf['test_predicted_Open']],
              labels={'value':'Stock price','Date': 'Date'})
fig.update_layout(title_text='Comparision between original Open price vs predicted Open price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Open Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[42]:


len(test_data)


# In[43]:


x_input=test_data[862:].reshape(1,-1)
x_input.shape


# In[44]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[45]:


temp_input


# In[46]:


# demonstrating prediction for next 10 days
from numpy import array

lst_output=[]
n_step=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_step,1))
        
        yhat=model.predict(x_input,verbose=0)
        print("{} day input {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_step,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
        
print(lst_output)
    


# In[47]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[48]:


len(df1)


# In[49]:


df3=df1.tolist()
df3.extend(lst_output)


# In[50]:


plt.plot(day_new,Scaler.inverse_transform(df1[3106:]))
plt.plot(day_pred,Scaler.inverse_transform(lst_output))


# In[52]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[3100:])


# In[61]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# In[62]:


last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)


# In[63]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = Scaler.inverse_transform(df1[len(df1)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = Scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})
names = cycle(['Last 100 days close price','Predicted next 30 days close price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 100 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# ![Screenshot%202023-11-20%20121428.png](attachment:Screenshot%202023-11-20%20121428.png)

# #### Insight
# -  The final output showns the increasing trend.
# -  Original share price also has increasing trend. 

# In[56]:


lstmgrudf=df1.tolist()
lstmgrudf.extend((np.array(lst_output).reshape(-1,1)).tolist())
lstmgrudf=Scaler.inverse_transform(lstmgrudf).reshape(1,-1).tolist()[0]

names = cycle(['Close price'])

fig = px.line(lstmgrudf,labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

