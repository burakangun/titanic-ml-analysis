import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



data = pd.read_csv('train.csv')

#Viewing data and different features
print(data.head())

print(data.shape)

print(data['Sex'].value_counts())


# Visualizing survivals based on gender
# We can see that females are survived most
data['Died'] = 1 - data['Survived']
data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar',
                                                           figsize=(10, 5),
                                                           stacked=True)

df1=data.drop(['Name','Ticket','Cabin','PassengerId','Died'], axis=1)


print(df1.head(10))

print('Null datas : ', data.isnull().sum())

df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df1.head()

# Fill NaN's in the Age column.
mean_age_men=data[data['Sex']==1]['Age'].mean()
mean_age_women=data[data['Sex']==0]['Age'].mean()

#Filling all the null values in 'Age' with respective mean age
data.loc[(data['Age'].isnull()) & (data['Sex']==0),'Age']=mean_age_women
data.loc[(data['Age'].isnull()) & (data['Sex']==1),'Age']=mean_age_men

df1.Age = (df1['Age']-min(df1['Age']))/(max(df1['Age'])-min(df1['Age']))
df1.Fare = (df1['Fare']-min(df1['Fare']))/(max(df1['Fare'])-min(df1['Fare']))
df1 = df1.dropna()



#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df1.drop(['Survived'], axis=1),
    df1.Survived,
    test_size= 0.2,
    random_state=0,
    stratify=df1.Survived)



rfr = RandomForestClassifier()
rfr.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_predict = rfr.predict(X_test)
print("Accuracy is : ",round(accuracy_score(y_test, y_predict),2))


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cma=confusion_matrix(y_test, y_predict)

# Matplotlib kullanarak ısı haritası çiz
plt.figure(figsize=(8, 6))
plt.imshow(cma, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# X ve Y eksenine etiket ekle
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test))
plt.yticks(tick_marks, np.unique(y_test))

# Anotasyonları ekle
thresh = cma.max() / 2.
for i, j in np.ndindex(cma.shape):
    plt.text(j, i, format(cma[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cma[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Grafiği göster
plt.show()



# Now we will predict the persons whether they live or not by loading the dataset test.csv

data_test = pd.read_csv('test.csv')

#Viewing test data
data_test.head()

df2=data_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

#Converting the categorical features 'Sex' and 'Embarked' into numerical values 0 & 1
df2['Sex']=df2['Sex'].map({'female':0, 'male':1})
df2['Embarked']=df2['Embarked'].map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df2.head()

#Let's check for the null values
df2.isnull().sum()

#Finding mean age
mean_age_men2=df2[df2['Sex']==1]['Age'].mean()
mean_age_women2=df2[df2['Sex']==0]['Age'].mean()

#Filling all the null values in 'Age' and 'Fare' with respective mean age and mean fare
df2.loc[(df2['Age'].isnull()) & (df2['Sex']==0),'Age']=mean_age_women2
df2.loc[(df2['Age'].isnull()) & (df2['Sex']==1),'Age']=mean_age_men2
df2['Fare']=df2['Fare'].fillna(df2['Fare'].mean())

df2.isnull().sum()

#Doing Feature Scaling to standardize the independent features present in the data in a fixed range
df2.Age = (df2['Age']-min(df2.Age))/(max(df2['Age'])-min(df2.Age))
df2.Fare = (df2['Fare']-min(df2.Fare))/(max(df2['Fare'])-min(df2.Fare))
df2.describe()

prediction = rfr.predict(df2)

submission = pd.DataFrame({"PassengerId": data_test["PassengerId"],
                            "Survived": prediction})
submission.to_csv('submission.csv', index=False)

prediction_df = pd.read_csv('submission.csv')

#Visualizing predicted values
# Count the occurrences of each value in the 'Survived' column
count_values = prediction_df['Survived'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(count_values.index, count_values.values, color=['blue', 'orange'])
plt.title('Count of Predicted Values')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks(count_values.index, ['Not Survived', 'Survived'])
plt.show()



