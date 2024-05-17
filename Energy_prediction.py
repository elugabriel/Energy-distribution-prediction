#!/usr/bin/env python
# coding: utf-8

# ## Import Basic Libraries

# In[1]:


import pandas as pd


# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense


# In[4]:


from tensorflow.keras.models import Sequential


# In[5]:


from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[8]:


data = pd.read_excel("abuja-weather-data.xlsx")


# In[9]:


data.rename(columns = {"Energy Consumption (GWh)": "Energy"}, inplace=True)


# In[10]:


data.head()


# # Data Preprocessing

# In[11]:


data.info()


# In[12]:


data['total_sunshine_hours'] = (data['sunset'] - data['sunrise']).dt.total_seconds() / 3600


# In[13]:


data.head()


# In[14]:


data = data.drop(["sunrise", "sunset", "temp", "region"], axis=1)


# In[15]:


data.head()


# In[16]:


data.describe()


# In[17]:


data.isnull().sum()


# In[18]:


data.isna().sum()


# In[19]:


data.head()


# ## Data Visualization

# In[20]:


# @title pressure

from matplotlib import pyplot as plt
data['pressure'].plot(kind='line', figsize=(8, 4), title='pressure')
plt.grid()
plt.gca().spines[['top', 'right']].set_visible(False)


# In[21]:


# @title temp_min

from matplotlib import pyplot as plt
data['temp_min'].plot(kind='line', figsize=(8, 4), title='temp_min')
plt.grid()
plt.gca().spines[['top', 'right']].set_visible(False)


# In[22]:


# @title description

from matplotlib import pyplot as plt
import seaborn as sns
data.groupby('description').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.grid()
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[23]:


# @title pressure

from matplotlib import pyplot as plt
data['pressure'].plot(kind='hist', bins=20, title='pressure')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[24]:


# @title temp_max

from matplotlib import pyplot as plt
data['temp_max'].plot(kind='hist', bins=20, title='temp_max')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[25]:


# @title temp_min

from matplotlib import pyplot as plt
data['temp_min'].plot(kind='hist', bins=20, title='temp_min')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[26]:


from matplotlib import pyplot as plt
data['temp_max'].plot(kind='hist', bins=20, title='Energy Consumption (GWh)')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[27]:


# @title temp_min

from matplotlib import pyplot as plt
data['temp_min'].plot(kind='hist', bins=20, title='total_sunshine_hours')
plt.gca().spines[['top', 'right',]].set_visible(False)


# # Data Manipulation

# In[28]:


data.head()


# In[29]:


data.description.unique()


# In[30]:


one_hot_encoded_data = pd.get_dummies(data, columns=['description']).astype(float)


# In[31]:


one_hot_encoded_data.head()


# In[32]:


one_hot_encoded_data.shape


# In[33]:


data = one_hot_encoded_data.copy()


# ## Data Spliting

# In[34]:


from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(one_hot_encoded_data, test_size=0.4, random_state=42)


# In[35]:


test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)


# # Data Normalization / Scaling

# In[36]:


train_stat = train_data.describe()
train_stat.pop("Energy")
train_stat = train_stat.transpose()
train_stat


# In[37]:


train_label = train_data.pop("Energy")
test_label = test_data.pop("Energy")
val_data_label = val_data.pop("Energy")


# In[38]:


def norm(x):
    return (x- train_stat['mean']) / train_stat['std']


norm_train_data = norm(train_data)
norm_test_data = norm(test_data)
norm_val_data = norm(val_data)


# In[39]:


print(r'Train/Test/Validate split: ')
print(f'Train:    {norm_train_data.shape}')
print(f'Test:     {norm_test_data.shape}')
print(f'Validate: {norm_test_data.shape}')


print(r'Train/Test/Validate Labels: ')
print(f'Train Labels:    {train_label.shape}')
print(f'Test Labels:     {test_label.shape}')
print(f'Validate Labels: {val_data.shape}')


# In[40]:


norm_train_data.head()


# # **Build Artificial Neural Network Model**

# In[41]:


def build_model():
    model = Sequential()
    
    # Define the neural network architecture
    model.add(Dense(32, input_shape = (norm_train_data.shape[1], )))
    
    model.add(Dense(64, Activation('relu')))
    
    model.add(Dense(128, Activation('relu')))
    
    model.add(Dense(32, Activation('relu')))
    
    model.add(Dense(1))
    
    # Compile the model
    learning_rate = 0.001
    optimizer = 'adam'
    model.compile(loss='mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    
    return model


# In[42]:


model = build_model()
print("Here is the Model Summary")
model.summary()


# # Model Trainning

# In[43]:


# Define epochs and batch size
epochs = 500
batch_size = 32
import time
# Build the model
model = build_model()
print("This is the model Summary")
model.summary()

# Train the model
start_time = time.time()
history = model.fit(norm_train_data,
                    train_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    shuffle=True,
                    steps_per_epoch=int(norm_train_data.shape[0] / batch_size),
                    validation_data=(norm_val_data, val_data_label)
)
end_time = time.time()
training_time = end_time - start_time
print("Training time:", training_time, "seconds")


# In[44]:


print(history)


# In[45]:


# Loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Metrics
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('Model Metrics')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()


# In[46]:


model.fit(norm_train_data,train_label, validation_data = (norm_val_data, val_data_label),epochs=150, batch_size=20)


# In[ ]:




