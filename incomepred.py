#import libraries
import pandas as pd
import numpy as np

#import datasets
TrainData = pd.read_csv('TrainData.csv')
TestData = pd.read_csv('TestData.csv')

#Delete columns which do not directly influence income
TrainData = TrainData.drop(columns = ['Instance', 'Wears Glasses', 'Body Height [cm]', 'Hair Color'] )
TestData = TestData.drop(columns = ['Instance', 'Wears Glasses', 'Body Height [cm]', 'Hair Color'] )
TrainData = TrainData[TrainData['Income in EUR'] > 0] 

#Handling missing data
TrainData = TrainData.fillna(TrainData.mean())
TestData = TestData.fillna(TestData.mean())

TrainData = TrainData.fillna(method = "ffill")
TestData = TestData.fillna(method = "ffill")

Train_gender = TrainData.Gender
Test_gender = TestData.Gender
Train_gender = Train_gender.replace('0', 'unknown')
Test_gender = Test_gender.replace('0', 'unknown')

Train_uni = TrainData['University Degree']
Test_uni = TestData['University Degree']
Train_uni = Train_uni.replace('0', 'No')
Test_uni = Test_uni.replace('0', 'No')

TrainData.Gender = Train_gender
TestData.Gender = Test_gender
TrainData['University Degree'] = Train_uni
TestData['University Degree'] = Test_uni

#Split train and test set
X_train = TrainData.iloc[:,:-1]
y_train = TrainData.iloc[:,7].values
X_test = TestData.iloc[:,:-1]

#Add a column in the end of each train set to easily segregate them post concatenation and encoding 
X_train['train'] = 1
X_test['train'] = 0
combined = pd.concat ([X_train,X_test]).reset_index(drop = True)

#convert Profession text to token counts
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vect = vectorizer.fit_transform(combined.Profession)

count_vect = pd.DataFrame(vect.todense(), columns = vectorizer.get_feature_names())
combined = pd.concat([combined, count_vect], axis = 1)
combined = combined.drop(columns = ['Profession'])

#Applying label encoder followed by one hot encoder
combined1 = combined.values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encode = LabelEncoder()
combined1[:,1] = encode.fit_transform(combined1[:,1])
combined1[:,3] = encode.fit_transform(combined1[:,3])
combined1[:,5] = encode.fit_transform(combined1[:,5])

hotencode = OneHotEncoder(categorical_features=[1])
combined1 = hotencode.fit_transform(combined1).toarray()
hotencode = OneHotEncoder(categorical_features=[6])
combined1 = hotencode.fit_transform(combined1).toarray()
hotencode = OneHotEncoder(categorical_features=[173])
combined1 = hotencode.fit_transform(combined1).toarray()

#Separate train and test sets and delete the column added to identify them
combined2 = pd.DataFrame(combined1)

df1 = combined2.loc[combined2 [177] == 1]
df2 = combined2.loc[combined2 [177] == 0]
del df1[177]
del df2[177]

trainx = df1.values
testx = df2.values

#Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
trainx = scaler.fit_transform(trainx)
testx = scaler.transform(testx)

#import keras library
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

#create model
model = Sequential()
model.add(Dense(12, input_dim=1180, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#compile model
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

#evaluate model and predict
model.fit(trainx, y_train, epochs=300, batch_size=143,  verbose=1)
y_pred = model.predict(testx)

