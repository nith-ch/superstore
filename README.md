# Introduction
This is a superstore dataset, a kind of simulation where show extensive data analysis to give insights on 
how the business can improve its earnings while minimizing the losses.

# Importing the libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
```

## Load dataset
```
Superstore = pd.read_csv('C:\\Users\\User\\Desktop\\GIT_Projects\\Pyyhon_Practice\\
SampleSuperstore\\SampleSuperstore.csv', index_col=False)
```
```
Superstore
```

## Check NULL
```
Superstore.isnull().sum()
```
<img src="https://raw.githubusercontent.com/nith-ch/superstore/master/pic/check_null.png" height="372" width="408">

## Check types
```
Superstore.info()
```
<img src="https://raw.githubusercontent.com/nith-ch/superstore/master/pic/info.png" height="372" width="408">

Superstore.head()
```

Superstore.shape
```
<img src="https://raw.githubusercontent.com/nith-ch/superstore/master/pic/shape.png" height="52" width="248">

## Check Duplicate Records
```
Superstore.duplicated().sum()
Superstore.drop_duplicates(inplace= True)

print(set(Superstore['State'].unique()))
```

## Remove unwanted columns
```
Superstore.drop(["Postal Code"],axis=1, inplace = True)
Superstore
```

## Unit Sales calculation
```
Superstore['UnitPrice'] = Superstore.Sales / Superstore.Quantity
```

## Unit Profit calculation
```
Superstore['UnitProfit'] = Superstore.Profit / Superstore.Quantity 
```

```
sns.pairplot(Superstore,hue='Ship Mode')
```

## Check Correlation of the data
```
Superstore.corr()
sns.heatmap(Superstore.corr(), annot=True)
```

## Segment and Category count
```
Superstore["Segment"].value_counts()
Superstore["Category"].value_counts()
```

## Category Bar Count
```
sns.countplot(x= Superstore['Category'])
```

## Distribution of Ship Mode
```
fig = plt.figure(figsize=(7,10))
```
## Check the most item superstore that has been shipped
```
ax1 = fig.add_subplot(212)
sns.countplot(y= Superstore['Ship Mode'], palette='deep')
ax1.set_title('Distribution of superstore by Ship Mode', loc='center')
plt.show()
```

## Box Plot of Sales by Sub-Category
```
fig = plt.figure(figsize=(20,10))
sns.boxplot(x='Sub-Category', y='Sales', data=Superstore)
plt.title('Box Plot of Sales by Sub-Category')
plt.show()
```

## Factors with Superstore sales
```
fig = px.scatter(Superstore,x="Profit",y="Sales",color="Discount",
                 size="Quantity",symbol="Ship Mode",title="How diffrent factors affects Superstore's sales ")
fig.update_layout(height=500, width=700,
                  legend=dict(yanchor="top", y=0.99, 
                              xanchor="left", x=0.01))
fig.show()
```

## Scatter Plot of Sales and Profit by Discount
```
Superstore.plot(kind="scatter",x="Sales",y="Profit", c="Discount", colormap="Set1",figsize=(15,10))
plt.title('Scatter Plot of Sales and Profit by Discount')
```

## Region Wise Profit And Segment
```
data1= Superstore.groupby("Segment")[["Sales","Profit"]].sum().sort_values(by="Sales", ascending=True)
data1[:].plot.area(color = ["lightgreen","darkgreen"], figsize=(10,7))
plt.title("Profit and Sales across Segment")
plt.show()
```

## Check Distribution of Sales (Low-value data is more frequent than high-value data)
```
plt.figure(figsize=(8,8))
plt.title('Superstore Sales Distribution Plot')
sns.distplot(Superstore['Sales']);
```

## Check Distribution of Profit (Two sides are of the same frequency)
```
plt.figure(figsize=(8,8))
plt.title('Superstore Sales Distribution Plot')
sns.distplot(Superstore['Profit']);
```
```
Superstore.loc[:,['Sales','Profit']].values
```

## We take just the Sales and Profit
```
Superstore1=Superstore[["Ship Mode","Segment","Region","Category","Sub-Category",
"Discount","Sales","Profit","UnitPrice","UnitProfit"]]
X=Superstore1[["Sales","Profit"]]
X_log = np.log(X)
X.head()
```

## Scatter Plot of Sales and Profit by Ship Mode
```
fig = plt.figure(figsize=(15,10))
sns.scatterplot(x=Superstore1['Sales'],y=Superstore1['Profit'], hue=Superstore1['Ship Mode'])
```

## Check the number of clusters
```
Cluster=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X)
    Cluster.append(km.inertia_)
```
## The elbow curve
```
plt.figure(figsize=(12,6))
plt.plot(range(1,11),Cluster)
plt.plot(range(1,11),Cluster, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("Cluster")
plt.show()
```

## 4 clusters
```
km1=KMeans(n_clusters=4)
```
## Fitting the data
```
km1.fit(X)
```
## predicting the labels of the data
```
y=km1.predict(X)
```
## Adding the labels to a column named label
```
Superstore1["Labels"] = y
```
## The new dataframe with the clustering
```
Superstore1.head()
```

```
data = X.assign(ClusterLabel = km1.labels_)
data.groupby("ClusterLabel")[["Sales", "Profit"]].median()
```
