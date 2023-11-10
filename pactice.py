from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns


# 1.Perform descriptive analysis and identify the data type
df=pd.read_csv("student.csv")
print(df.head())
print(df.describe())
print(df.info())



# 2.Implement a method to find out variation in data. For example the difference between highest and lowest marks in each subject semester wise
mean1=df["marks"].mean()
median=df["marks"].median()
mode=df["marks"].mode()
q1=df["marks"].quantile(0.25)
q3=df["marks"].quantile(0.75)
iqr=q3-q1
mini=df["marks"].min()
maxi=df["marks"].max()
range1=maxi-mini
print("mean:",mean1," median:",median," mode:", mode," iqr:",iqr, "range:",range1)



# 3.Plot the graph showing the result of students in each semester.
# groupd= df.groupby("Semester")
# for marks, group in groupd:
    # plt.bar( group["Semester"],group["marks"])
# plt.show()



# 4. Plot the graph showing the geographical location of students.
geolocator=Nominatim(user_agent="student_locator")
def get_coordinates(city):
    location=geolocator.geocode(city)
    return location.latitude, location.longitude
df['Coordinates'] = df['city'].apply(get_coordinates)
df[['Latitude', 'Longitude']] = pd.DataFrame(df['Coordinates'].tolist(), index=df.index)
df.drop(columns=['Coordinates'], inplace=True)
print(df)
# plt.figure(figsize=(10, 6))
# plt.scatter(df['Longitude'], df['Latitude'], c='red', s=100)
# for i, row in df.iterrows():
    # plt.text(row['Longitude'], row['Latitude'], row['city'])
# plt.show()



# 5. Plot the graph showing the number of male and female students.
# genderval=df["Gender"].value_counts()
# plt.bar(genderval.index, genderval.values)
# plt.show()



# 6. Implement a method to treat missing value for gender and missing value for marks.
print("missing values in gender before treatment:")
print(df["Gender"].isnull().value_counts())
print("missing values in gender before treatment:")
print(df["Gender"].isnull().value_counts())
max=df["Gender"].value_counts().idxmax()
df["Gender"].fillna(max, inplace=True)
mean=df["marks"].mean()
df["marks"].fillna(mean, inplace=True)
print("missing values in gender after treatment:")
print(df["Gender"].isnull().value_counts())
print("missing values in gender after treatment:")
print(df["Gender"].isnull().value_counts())



# 7. Linear regression
model=LinearRegression()
x=df["years"].values.reshape(-1, 1)
y=df["salary"]
xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.2,random_state=10)
model.fit(xtrain, ytrain)
ypred=model.predict(xtest)
trainscore=model.score(xtrain, ytrain)
testscore=model.score(xtest, ytest)
print("R-squared score on training data:", trainscore)
print("R-squared score on test data:", testscore)
# sns.scatterplot(x=xtest.flatten(), y=ytest, color="blue", label="actual")
# sns.lineplot(x=xtest.flatten(), y=ypred, color="green", label="predicted")
# plt.show()

# 8. Logistic regression
model1=LogisticRegression()
x1=df[["marks","marks1"]]
y1=df["status"]
xtrain1, xtest1, ytrain1, ytest1= train_test_split(x1,y1,test_size=0.2,random_state=10)
model1.fit(xtrain1,ytrain1)
ypred1=model1.predict(xtest1)
accuracy = accuracy_score(ytest1, ypred1)
print(f"Accuracy: {accuracy * 100:.2f}%")
x_min, x_max = x1['marks'].min() - 1, x1['marks'].max() + 1
y_min, y_max = x1['marks1'].min() - 1, x1['marks1'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x1['marks'], x1['marks1'], c=y1, marker='o', edgecolor='k')
plt.title("Logistic Regression: Clever (1) vs. Average (0)")
plt.xlabel("Marks1")
plt.ylabel("Marks2")
plt.show()



