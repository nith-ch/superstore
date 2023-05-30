# Superstore
```import numpy as np
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
Superstore = pd.read_csv('C:\\Users\\User\\Desktop\\GIT_Projects\\Pyyhon_Practice\\SampleSuperstore\\SampleSuperstore.csv', index_col=False)
```

## Check NULL
```
Superstore.isnull().sum()
```
