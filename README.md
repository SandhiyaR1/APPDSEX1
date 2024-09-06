# APPDSEX1
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

## CODING AND OUTPUT:
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
## RESULT:
Thus Data analysis and Data preprocessing implemeted using a dataset.
