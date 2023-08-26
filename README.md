# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
from google.colab import files
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv('/content/data.csv')
print(df)
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
print(df.isnull().sum())
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
Y= df.iloc[:,-1].values
print(Y)
df.duplicated()
print(df['Calories'].describe())
scaler= MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```
## OUTPUT:
![Screenshot 2023-08-26 093109](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/43b81178-e82e-414a-821b-ca1dd5739e84)
![Screenshot 2023-08-26 093346](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/6db50bc3-5023-4400-984a-6cdaf21bbb3b)
![Screenshot 2023-08-26 093404](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/9a70d899-debd-4223-8902-718c3d0d09a7)
![Screenshot 2023-08-26 093418](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/86bba68a-2342-4cb5-8aaa-9a1b387b9a87)
![Screenshot 2023-08-26 093428](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/ec68c858-3c60-4dfe-98ac-80a147520e2f)
![Screenshot 2023-08-26 093436](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/e8f65120-ec54-41b1-997d-b65dbf6f38d2)
![Screenshot 2023-08-26 093445](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/cd39348b-eaf7-4639-a1fa-733f86737859)
![Screenshot 2023-08-26 093455](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/d8415c98-eea3-41ad-8ffe-1b1e4d6a2894)
![Screenshot 2023-08-26 093535](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/297a4db3-a64a-4a8f-a9fd-e395a8d586c2)
![Screenshot 2023-08-26 093615](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/bc558394-9600-43c7-afb7-65208511e7b6)
![Screenshot 2023-08-26 093644](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/17d7af8a-d35a-4fa4-8519-ffc994e0d225)
![Screenshot 2023-08-26 093655](https://github.com/s-adhithya/Ex.No.1---Data-Preprocessing/assets/113497423/34899208-61dc-478e-b4ae-ca858660826b)


## RESULT
THe program executed successfully.
