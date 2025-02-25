#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError
import matplotlib.pyplot as plt


# In[3]:


# Load CSV data
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("test.csv")
lookup_data = pd.read_csv("IdLookupTable.csv")

# Drop rows with null values in the training data
train_data.dropna(inplace=True)

# Separate features and target labels in the training data
X = train_data['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values
X = np.array([x.reshape(96, 96, 1) for x in X]) / 255.0  # Normalize pixel values

# Extract keypoint labels and convert to numpy array
y = train_data.drop(['Image'], axis=1).values

# Reshape test images similarly
X_test = test_data['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values
X_test = np.array([x.reshape(96, 96, 1) for x in X_test]) / 255.0


# In[4]:


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.1),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(30)  # Output layer with 30 nodes (for each facial keypoint coordinate)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])
    return model

model = build_model()


# In[5]:


# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=64, 
                    callbacks=[early_stopping], verbose=1)


# In[6]:


# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Training and Validation MAE
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Mean Absolute Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Model Evaluation
train_loss, train_mae = model.evaluate(X, y)
print(f'Training Loss: {train_loss:.4f}')
print(f'Training Mean Absolute Error: {train_mae:.4f}')


# In[ ]:




