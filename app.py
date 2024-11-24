import pandas as pd
HDNames = {'Cement', 'BFS', 'FLA', 'Water', 'SP', 'CA', 'FA', 'Age', 'CCS'}
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd

file_path = '/content/drive/MyDrive/Colab Notebooks/ConcreteData.xlsx'
Data = pd.read_excel(file_path, names=HDNames)
columns_order = ['Cement', 'BFS', 'FLA', 'Water', 'SP', 'CA', 'FA', 'Age', 'CCS']
Data = Data[columns_order]
print(Data.head(20))
import seaborn as sns
sns.set(style="ticks")
sns.boxplot(data=Data)
sns.pairplot(data=Data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(Data))from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(Data))
DataScaled = scaler.fit_transform(Data)
# Convert the set to a list
HDNames = list(HDNames)
DataScaled = pd.DataFrame(DataScaled, columns=HDNames)
summary = DataScaled.describe()
print(summary)
sns.boxplot(data=DataScaled)
from sklearn.model_selection import train_test_split
Predictors = pd.DataFrame(DataScaled.iloc[:,:8])
Response = pd.DataFrame(DataScaled.iloc[:,8])
Pred_train, Pred_test, Resp_train, Resp_test = train_test_split(Predictors, Response, test_size=0.30, random_state=1)
print(Pred_train.shape)
print(Pred_test.shape)
print(Resp_train.shape)
print(Resp_test.shape)
from keras.models import Sequential
from keras.layers import Dense

# Define the model
model = Sequential()

# Add layers to the model
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(Pred_train, Resp_train, epochs=1000, verbose=1)
model.summary()
from sklearn.metrics import r2_score
y_PredKM = model.predict(Pred_test)
print("Coefficient of determination of keras model")
print(r2_score(Resp_test, y_PredKM))
Q1 = DataScaled.quantile(0.25)
Q3 = DataScaled.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
DataScaledOut = DataScaled[~((DataScaled < (Q1 - 1.5 * IQR)) |(DataScaled > (Q3 + 1.5 * IQR))).any(axis=1)]
print(DataScaledOut.shape)
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(121)
sns.boxplot(data=DataScaled)
plt.subplot(122)
sns.boxplot(data=DataScaledOut)
