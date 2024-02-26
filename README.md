# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This dataset presents a captivating challenge due to the intricate relationship between the input and output columns. The complex nature of this connection suggests that there may be underlying patterns or hidden factors that are not readily apparent.

## Neural Network Model

![image](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/7788db31-e1b0-4858-8111-4f0fd5f33284)


## DESIGN STEPS

## Step 1: Loading the Dataset
1. Load the dataset containing features and target variables into memory.
2. Check for data consistency and handle any missing values or anomalies.

## Step 2: Splitting the Dataset
1. Divide the dataset into training and testing subsets, ensuring a representative distribution of data in each subset.
2. Shuffle the data before splitting to avoid any inherent ordering bias.

## Step 3: Data Normalization
1. Normalize the features using MinMaxScaler to scale them within a predefined range, typically [0, 1].
2. Fit the scaler to the training data and transform both training and testing data accordingly.

## Step 4: Building the Neural Network Model
1. Design the architecture of the neural network model, specifying the number of layers and neurons per layer.
2. Compile the model by defining the loss function, optimizer, and any additional metrics to monitor during training.

## Step 5: Training the Model
1. Train the neural network model using the training data, specifying the number of epochs and batch size.
2. Monitor the training process for convergence and potential overfitting by observing the loss on both training and validation data.

## Step 6: Plotting Performance
1. Visualize the training process by plotting the training and validation loss over epochs.
2. Plot any additional metrics such as accuracy or precision to assess the model's performance.

## Step 7: Evaluating the Model
1. Evaluate the trained model's performance using the testing data.
2. Compute relevant metrics such as accuracy, precision, recall, and F1-score to assess the model's effectiveness.

## PROGRAM
### Name: SANTHOSH U
### Register Number: 212222240092
#### DEPENDENCIES:
```py
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```
#### DATA FROM SHEETS:
```py
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
#### DATA VISUALIZATION:
```py
import pandas as pd
import seaborn as sns
df['Input 1 (Number)'] = pd.to_numeric(df['Input 1 (Number)'])
sns.pairplot(df)

df['Input 1 (Number)'] = pd.to_numeric(df['Input 1 (Number)'])
df['Output'] = pd.to_numeric(df['Output'])
X = df['Input 1 (Number)']
y=df['Output']
```
#### DATA SPLIT AND PREPROCESSING:
```PY
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
M = MinMaxScaler()
x_train = M.fit_transform(x_train)
```
#### REGRESSIVE MODEL:
```py
model = Sequential()
model.add(Dense(15,activation='relu',input_shape=x_train.shape))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x_train,y_train,epochs=80)
model.history
```
#### LOSS CALCULATION:
```py
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
```
#### PREDICTION:
```py
y_pred=model.predict(x_test)
y_pred
```

## Dataset Information

![image](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/952c35cb-79d1-453d-9780-e211a080b1b6)


## OUTPUT
### Pairplot(data)
![download](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/c4937b65-3e29-4f5e-8cce-e2925522a8ab)

### ARCHITECTURE AND TRAINING:
![image](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/1a82aef8-a035-48a3-9c93-e8908b00e92f)

![image](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/c188bf64-7126-4b98-a1d7-82d83b7e76c2)

### Training Loss Vs Iteration Plot
![download](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/469a0bd5-2e56-4c2a-9544-f1bf1b54afbf)

### Test Data Root Mean Squared Error

![image](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/8eef4424-1609-417f-aa54-b0d3e49fddd6)

### New Sample Data Prediction

![image](https://github.com/aldrinlijo04/basic-nn-model/assets/118544279/41fed62c-2976-4ae8-8d0f-f82bb310f133)


## RESULT

Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
