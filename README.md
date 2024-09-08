# APPLIED-DATASCIENCE-EX1
Implementing Data Preprocessing and Data Analysis

## AIM:
To implement Data analysis and data preprocessing using a data set

## ALGORITHM:
Step 1: Import the data set necessary

Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.

Step 3: Perform Categorical data analysis.

Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.

Step 5: Implement Quantile transfomer to make the column value more normalized.

Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.

## CODING:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/content/Life Expectancy Data CSV.csv')
df.head()
df.info()
df.isnull().sum()
numerical_columns=df.select_dtypes(include=['number']).columns
numerical_columns
columns_to_fill = [
    'Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B',
    ' BMI ', 'Polio', 'Total expenditure', 'Diphtheria ',
    'GDP', 'Population', 'Income composition of resources', 'Schooling'
]
df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].median())

df.isnull().sum()
for column in numerical_columns:
  plt.figure(figsize=(8,6))
  plt.boxplot(df[column])
  plt.title(f'Boxplot of {column}')
  plt.show()
import pandas as pd
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

df_cleaned = df.copy()

for column in numerical_columns:
    Q1 = df_cleaned[column].quantile(0.25)
    Q3 = df_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]

print("Shape of DataFrame after outlier removal:", df_cleaned.shape)
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)
for column in categorical_columns:
    print(f"Value counts for {column}:")
    print(df_cleaned[column].value_counts())
    print("\n")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GDP', y='Life expectancy ', data=df_cleaned)
plt.title('Scatter Plot: Life expectancy vs GDP')
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(x='Year', hue='Status', data=df_cleaned)
plt.title('Count Plot: Year vs Status')
plt.xticks(rotation=90)
plt.show()
# Grouped box plot: 'Life expectancy' by 'Status' and 'Year'
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Life expectancy ', hue='Status', data=df_cleaned)
plt.title('Life Expectancy by Year and Status')
plt.xticks(rotation=90)
plt.show()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_cleaned['Country']=le.fit_transform(df_cleaned['Country'])
!pip install category_encoders
from category_encoders import BinaryEncoder
be=BinaryEncoder()
nbe=be.fit_transform(df_cleaned['Status'])
df_cleaned=pd.concat([df_cleaned,nbe],axis=1)
df_cleaned.drop(columns=['Status'],inplace=True)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
columns_to_scale=['Life expectancy ','infant deaths', 'Alcohol', 'Hepatitis B',
       ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', 'GDP', 'Income composition of resources',
       'Schooling']
df_cleaned[columns_to_scale]=scaler.fit_transform(df_cleaned[columns_to_scale])
from sklearn.preprocessing import RobustScaler
rscaler=RobustScaler()
columns_to_rscaler=['Adult Mortality', 'percentage expenditure','Measles ', 'Population']
df_cleaned[columns_to_rscaler]=rscaler.fit_transform(df_cleaned[columns_to_rscaler])
df_cleaned.head()
corr_matrix = df_cleaned.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
selected_columns = ['Life expectancy ', 'GDP', 'Alcohol', ' BMI ', 'Schooling']
sns.pairplot(df_cleaned[selected_columns])
plt.suptitle('Pairplot of Selected Numerical Columns', y=1.02)
plt.show()
```
## OUTPUT:
### df.head()
![image](https://github.com/user-attachments/assets/015fed6d-cd94-4266-8e0a-ab6dfebbdb6a)
### df.info()
![image](https://github.com/user-attachments/assets/20d342ba-9ab9-404f-9c19-05b22bb3cec7)
### df.isnull().sum()
![image](https://github.com/user-attachments/assets/627495db-6f75-497f-9e3d-0a21d5880a7e)
### Numerical_columns
![image](https://github.com/user-attachments/assets/efa0d0d5-6a86-4df7-89d1-260127d47c3a)
### Afetr filling NULL values- df.isnull().sum()
![image](https://github.com/user-attachments/assets/29cc3dc3-16f4-43d0-9a3b-762ed540300c) 
### Boxplot to identify outliers
![image](https://github.com/user-attachments/assets/681c0685-2861-4571-90f2-1e2618e1c44a) ![image](https://github.com/user-attachments/assets/a7f8d166-543b-4092-a9bc-f8a4b44ba0f5) 
![image](https://github.com/user-attachments/assets/58814432-4ac6-4df2-894b-a72ce7434fb5)  ![image](https://github.com/user-attachments/assets/39d5f087-a6e6-4f5b-8288-b4b31dda6082) 
![image](https://github.com/user-attachments/assets/da663f5d-5d6e-4818-9c27-8e43bf17f1be)  ![image](https://github.com/user-attachments/assets/8dbd3736-c6dd-4fda-b592-c0331c84bc07)  
![image](https://github.com/user-attachments/assets/11ef2c8d-ebbe-49fa-aa2d-299a922c222b)  ![image](https://github.com/user-attachments/assets/8499a282-c3d1-4aa4-a0ff-c108664a0ffa)
![image](https://github.com/user-attachments/assets/304f0604-5636-4a53-913b-6d931a2c6365)  ![image](https://github.com/user-attachments/assets/f7454be8-cb9c-466f-b7b6-8791d90af0d8) 
![image](https://github.com/user-attachments/assets/0040678a-5c06-4f55-942d-d45fd40d8713)   ![image](https://github.com/user-attachments/assets/a845e2f5-26c9-4aa3-8249-af2cdde53528) 
![image](https://github.com/user-attachments/assets/76299585-d1dd-404f-a64b-bb5f9fa3a9a5)    ![image](https://github.com/user-attachments/assets/e568586a-5fab-4d85-9580-ec3e3db33552) 
![image](https://github.com/user-attachments/assets/0b7be52b-3bf1-4458-89de-272fbad9df7b)    ![image](https://github.com/user-attachments/assets/4c996610-947e-44da-ab71-cb4cd8dc1b50)
### outliers removed
![image](https://github.com/user-attachments/assets/de502440-3ff8-4ced-b86c-10a720df21e1)
### Value counts for Country and Status
![image](https://github.com/user-attachments/assets/bfcbd446-b3b9-48a3-aba9-6d463bff3cc7)
### Scatter Plot: Life expectancy vs GDP
![image](https://github.com/user-attachments/assets/b7270308-91d4-4521-89b0-7b8636c69f17)
### Count Plot: Year vs Status
![image](https://github.com/user-attachments/assets/3141ef02-84d7-4dec-968d-706f45f96217)
### Count Plot: Life Expectancy by Year and Status
![image](https://github.com/user-attachments/assets/0b44ef57-2a39-4110-b7f2-99c8eefc1225)
### cleaned data
![image](https://github.com/user-attachments/assets/372f11e4-ab45-4069-abcf-bf5048cb3ec9)
### Correlation Heatmap
![image](https://github.com/user-attachments/assets/68381809-47a4-445d-996f-a689b3908559)
### Pairplot of Selected Numerical Columns
![image](https://github.com/user-attachments/assets/8d32775a-850b-4e3f-a942-954f41ad83ec)
![image](https://github.com/user-attachments/assets/38e96f83-f430-4ca1-acd4-0e42d0303f06)

























## RESULT:
Thus Data analysis and Data preprocessing implemeted using a dataset.
