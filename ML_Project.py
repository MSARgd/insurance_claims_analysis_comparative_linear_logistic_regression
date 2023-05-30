#!/usr/bin/env python
# coding: utf-8

# <a id="I"></a>
# 
# # I.  Reading Data - Exploratory Data Analysis with Pandas

# In[80]:


# Soufiane MOUHTARAM M.Sid ABDELLAH regagde
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set()  #  Will import Seaborn functionalities
# we don't like warnings
# you can comment the following 2 lines if you'd like to
import warnings
warnings.filterwarnings('ignore')


# 
# We’ll demonstrate the main methods in action by analyzing a [dataset](https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383) on the churn rate of telecom operator clients. Let’s read the data (using `read_csv`), and take a look at the first 5 lines using the `head` method:
# 

# In[81]:


# Soufiane MOUHTARAM M.Sid ABDELLAH regagde
# Disply all Columns
pd.options.display.max_columns=40


# In[3]:


# Soufiane MOUHTARAM M.Sid ABDELLAH regagde
autinsurance = pd.read_csv('insurance_claims_complete.csv')
autinsurance.head(5)


# In[4]:


# Soufiane MOUHTARAM M.Sid ABDELLAH regagde
autinsurance.info()


# In[5]:


# Soufiane MOUHTARAM M.Sid ABDELLAH regagde
autinsurance.isna().sum()  ## Count Filled cells
                             ## Sum empty cells


# In[6]:


# Soufiane MOUHTARAM M.Sid ABDELLAH regagde

# Column names may be accessed (and changed) using the `.columns` attribute as below
print("Old Column Names:\n", autinsurance.columns) 


# In[7]:


# !pip install pyjanitor 
import janitor
autinsurance = autinsurance.clean_names()
print("New Janitor Column Names:\n", autinsurance.columns) 


# In[8]:


autinsurance.shape


# In[9]:


print(autinsurance.columns)


# In[10]:


autinsurance.info()


# In[11]:


autinsurance.describe() # help us to describe dataframe and give us the statistical values


# In order to see statistics on non-numerical features, one has to explicitly indicate data types of interest in the `include` parameter.

# In[12]:


autinsurance.describe(include=['object', 'bool'])


# For categorical (type `object`) and boolean (type `bool`) features we can use the `value_counts` method. Let’s have a look at the distribution of `fraud_reported`:

# In[13]:


autinsurance['fraud_reported'].value_counts()


# In[14]:


autinsurance['fraud_reported'].value_counts(normalize=True)


# In[15]:


(autinsurance['fraud_reported'].value_counts().plot(
        kind='bar',
        figsize=(8, 6),
        title='Distribution of Target Variable',
    )
);
plt.show()


# In[16]:


autinsurance['fraud_reported'].mean()


# In[17]:


autinsurance.columns


# <a id="II"></a>
# # II. Visual data analysis in Python
# 

# ### Article outline
# 
# 1. Dataset
# 2. Univariate visualization
#     * 2.1 Quantitative features
#     * 2.2 Categorical and binary features
# 3. Multivariate visualization
#     * 3.1 Quantitative–Quantitative
#     * 3.2 Quantitative–Categorical
#     * 3.3 Categorical–Categorical
# 4. Whole dataset
#     * 4.1 Naive approach
#     * 4.2 Dimensionality reduction
#     * 4.2 t-SNE
# 5. Useful resources

# ### 1. Univariate visualization
# 
# *Univariate* analysis looks at one feature at a time. When we analyze a feature independently, we are usually mostly interested in the *distribution of its values* and ignore other features in the dataset.
# 
# Below, we will consider different statistical types of features and the corresponding tools for their individual visual analysis.
# 
# #### 1.1 Quantitative features
# 
# *Quantitative features* take on ordered numerical values. Those values can be *discrete*, like integers, or *continuous*, like real numbers, and usually express a count or a measurement.
# 
# ##### 1.1.1 Histograms and density plots
# 
# The easiest way to take a look at the distribution of a numerical variable is to plot its *histogram* using the `DataFrame`'s method [`hist()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html).

# In[18]:


features = ['months_as_customer', 'capital_gains', 'capital_loss']

autinsurance[features].hist(figsize=(10, 10));


# In[19]:


autinsurance[features].plot(kind='density', subplots=True, layout=(2, 2), 
                  sharex=False, figsize=(10, 10));


# In[20]:


# increasing the width of the Chart
import seaborn as sns
plt.rcParams['figure.figsize'] = 10,10 # similar to par(mfrow = c(2,1), mar = c(4,4,2,1)) # 2 columns and 1 row
sns.distplot(autinsurance["age"]) # pass it one variable
# if you are getting warnings related to the package you should use ignore function
import warnings
warnings.filterwarnings ('ignore')


# In[21]:


# increasing the width of the Chart
plt.rcParams['figure.figsize'] = 4,4 # similar to par(mfrow = c(2,1), mar = c(4,4,2,1)) # 2 columns and 1 row
sns.distplot(autinsurance["months_as_customer"]) # pass it one variable

# if you are getting warnings related to the package you should use ignore function
import warnings
warnings.filterwarnings ('ignore')


# #### 1.2 Categorical and binary features
# 
# *Categorical features* take on a fixed number of values. Each of these values assigns an observation to a corresponding group, known as a *category*, which reflects some qualitative property of this example. *Binary* variables are an important special case of categorical variables when the number of possible values is exactly 2. If the values of a categorical variable are ordered, it is called *ordinal*.
# 
# ##### 1.2.1 Frequency table
# 
# Let’s check the class balance in our dataset by looking at the distribution of the target variable: the *churn rate*. First, we will get a frequency table, which shows how frequent each value of the categorical variable is. For this, we will use the [`value_counts()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html) method:

# In[22]:


autinsurance['fraud_reported'].value_counts()


# By default, the entries in the output are sorted from the most to the least frequently-occurring values.
# 
# In our case, the data is not *balanced*; that is, our two target classes, loyal and disloyal customers, are not represented equally in the dataset. Only a small part of the clients canceled their subscription to the telecom service. As we will see in the following articles, this fact may imply some restrictions on measuring the classification performance, and, in the future, we may want to additionaly penalize our model errors in predicting the minority "Churn" class.

# ##### 1.2.2 Bar plot
# 
# The bar plot is a graphical representation of the frequency table. The easiest way to create it is to use the `seaborn`'s function [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html). There is another function in `seaborn` that is somewhat confusingly called [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) and is mostly used for representation of some basic statistics of a numerical variable grouped by a categorical feature.
# 
# Let's plot the distributions for two categorical variables:

# In[23]:


_, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

sns.countplot(x='policy_state', data=autinsurance, ax=axes[0]);
sns.countplot(x='incident_state', data=autinsurance, ax=axes[1]);


# In[24]:


_, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

sns.countplot(x='insured_education_level', data=autinsurance, ax=axes[0]);
sns.countplot(x='insured_relationship', data=autinsurance, ax=axes[1]);


# #### 1.2.3. Distributions of categorical features

# In[25]:


_, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

sns.countplot(x='insured_sex', data=autinsurance, ax=axes[0]);
sns.countplot(x='incident_state', data=autinsurance, ax=axes[1]);


# In[26]:


# Distributions of categorical features
plt.rcParams['figure.figsize'] = 8,6
sns.countplot(y='insured_sex', data=autinsurance)
plt.show()

sns.countplot(y='incident_state', data=autinsurance)
plt.show()


# ### 2. Multivariate visualization
# 
# *Multivariate* plots allow us to see relationships between two and more different variables, all in one figure. Just as in the case of univariate plots, the specific type of visualization will depend on the types of the variables being analyzed.
# 
# #### 2.1 Quantitative–Quantitative
# 
# ##### 2.1.1 Correlation matrix
# 

# In[27]:


autinsurance.head(5)


# In[28]:


corr_matrix = autinsurance.corr(method = 'pearson')  # corr(autinsurance)
corr_matrix


# <h4>Highly correlated items = not good!</h4>
# <h4>Low correlated items = good </h4>
# <h4>Correlations with target (dv) = good (high predictive power)</h4>

# In[29]:


# seaborn
## first_twenty = har_train.iloc[:, :20] # pull out first 20 feats
corr = autinsurance.corr()  # compute correlation matrix
mask = np.zeros_like(corr, dtype=np.bool)  # make mask
mask[np.triu_indices_from(mask)] = True  # mask the upper triangle

fig, ax = plt.subplots(figsize=(11, 9))  # create a figure and a subplot
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # custom color map
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    center=0,
    linewidth=0.5,
    cbar_kws={'shrink': 0.5}
);


# In[30]:


autinsurance


# ##### How to calculate correlation between all columns and remove highly correlated ones using pandas?

# In[31]:


import numpy as np

# Create correlation matrix
corr_matrix = autinsurance.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

# Drop features 
autinsurance.drop(to_drop, axis=1, inplace=True)

autinsurance.shape


# In[32]:


# Univariate Histograms
plt.rcParams['figure.figsize'] = 20,20  # control plot size

autinsurance.hist()
plt.show()


# <a id="III"></a>
# # III. Visual Data Analysis in with Profile Report
# 

# In[34]:


import pandas_profiling
autinsurance.profile_report()


# In[82]:


# Import Libraries
import sweetviz as sv
import pandas as pd


# In[83]:


# Analyzing data
report=sv.analyze(autinsurance)

# Generating report
report.show_html('eda_report.html')


# #### Playing with Reports

# In[37]:


report.show_html(filepath='eda_report.html',
open_browser=True,
layout='vertical',
scale=0.7)


# In[38]:


report.show_notebook(w=None, 
                h=None, 
                scale=None,
                layout='vertical',
                filepath='sweetviz_report.html')


# In[39]:


autinsurance.columns


# ### Prior Knowledge for remove unuseful feautures

# In[40]:


to_delete= [ 'policy_number', 'policy_bind_date', 'insured_zip', 'incident_date', 'incident_city',
            'incident_location', 'incident_hour_of_the_day', 'total_claim_amount','injury_claim', 
            'property_claim', 'auto_year' ]


# In[41]:


autinsuranceV2 = autinsurance.drop(to_delete, axis = 1)


# In[42]:


print(autinsurance.shape)
print(autinsuranceV2.shape)


# Like you see after **Prior** **Knowledge** we lose 12 colmns

# In[43]:


# Save to file
autinsuranceV2.to_csv("insurance_claimsV2.csv", index = False)


# <a id="III"></a>
# # III- Data Pre-processing &  Preparation

# ### Data Transformation
# 
# There are some algorithms that can work well with categorical data, such as decision trees. But most machine learning algorithms cannot operate directly with categorical data. These algorithms require the input and output both to be in numerical form. If the output to be predicted is categorical, then after prediction we convert them back to categorical data from numerical data. Let's discuss some key challenges that we face while dealing with categorical data:

# ####  Fixing Special Characters

# In[84]:


autinsurance_copy = autinsuranceV2.copy()

_, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

sns.countplot(x='collision_type', data=autinsurance_copy, ax=axes[0]);

sns.countplot(x='police_report_available', data=autinsurance_copy, ax=axes[1]);

sns.countplot(x='property_damage', data=autinsurance_copy, ax=axes[2]);


# ###  2. Encoding Categorical Data!
# #### Label Encoding
# 
# This is a technique in which we replace each value in a categorical column with numbers from 0 to N-1. For example, say we've got a list of employee names in a column. After performing label encoding, each employee name will be assigned a numeric label. But this might not be suitable for all cases because the model might consider numeric values to be weights assigned to the data. Label encoding is the **best method to use for ordinal data**. The scikit-learn library provides LabelEncoder(), which helps with label encoding. Let's look at an exercise in the next section.

# Before doing the encoding, remove all the missing data. To do so, use the dropna() function, Select all the columns that are not numeric using the following code:

# In[45]:


data_column_category = autinsurance_copy.select_dtypes(exclude=[np.number]).columns
data_column_category


# ```
# data_column_category =[
#     'policy_csl', 
#     'insured_sex',
#     'insured_education_level', 
#     'insured_occupation', 
#     'insured_hobbies',
#     'insured_relationship', 
#     'incident_type',
#     'collision_type', 
#     'incident_severity', 
#     'authorities_contacted',
#     'incident_state', 
#     'property_damage', 
#     'police_report_available', 
#     'auto_make', 'auto_model']
# ```

# In[46]:


data_column_category =[
    'policy_csl', 
    'insured_sex',
    'insured_education_level', 
    'insured_occupation', 
    'insured_hobbies',
    'insured_relationship', 
    'incident_type',
    'collision_type', 
    'incident_severity', 
    'authorities_contacted',
    'incident_state', 
    'property_damage', 
    'police_report_available', 
    'auto_make', 'auto_model']


# In[47]:


#import the LabelEncoder class

from sklearn.preprocessing import LabelEncoder

#Creating the object instance

label_encoder = LabelEncoder()

for i in data_column_category:

    autinsurance_copy[i] = label_encoder.fit_transform(autinsurance_copy[i])

print("Label Encoded Data: ")

autinsurance_copy.head()


# In[48]:


autinsurance_copy.columns


# In[49]:


autinsurance_copy


# In[50]:


# Save to file
autinsurance_copy.to_csv("insurance_claimsV33.csv", index = False)


# In[51]:


# Disply all Columns
pd.options.display.max_columns=70


# In[52]:


autinsurance = pd.read_csv('insurance_claimsV33.csv')
autinsurance.head()


# <a id="III"></a>
# # Data Pre-processing &  Preparation

# ###  Dealing with Missing Values
# 
# #### Delete a column

# In[53]:


# Disply all Columns
pd.options.display.max_rows=170

# Finding missing values

autinsurance.isna().sum()


# In[54]:


# removing Null values
autinsurance = autinsurance.dropna()
autinsurance.info()


# In[55]:


autinsurance.isna().sum()


# #### Impute missing values 
# 
# Impute the numerical data of the age column with its mean. To do so, first find the mean of the column with missing values using the mean() function of pandas, and then print it  

# In[56]:


autinsurance.columns


# In[57]:


# Impute numerical data with mean '
mean_months_as_customer = autinsurance.months_as_customer.median()
print(mean_months_as_customer)


autinsurance.months_as_customer.fillna(mean_months_as_customer, inplace=True)


# In[58]:


autinsurance.head()


# ###  Finding and Fixing Outliers

# In[59]:


autinsurance.head()
# Rgagde Mohamed Sid Abdalla
X = autinsurance.drop(['policy_state'], axis=1).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = autinsurance['fraud_reported'].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
X_train,X_test ,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)


# ### Boxplot

# In[60]:


autinsurance['months_as_customer'].describe()


# In[61]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='months_as_customer', y='capital_gains', data=autinsurance)
plt.title("Scatter Plot of months_as_customer vs. capital-gains")
plt.show()


# In[62]:


# Calculate mean and standard deviation
mean = autinsurance['months_as_customer'].mean()
std_dev = autinsurance['capital_gains'].std()

# Calculate quartiles and interquartile range
Q1 = autinsurance['months_as_customer'].quantile(0.25)
Q3 = autinsurance['capital_gains'].quantile(0.75)
IQR = Q3 - Q1

# Z-Score
threshold = 2
autinsurance['z_score'] = np.abs((autinsurance['months_as_customer'] - mean) / std_dev)
outliers_zscore = autinsurance[autinsurance['z_score'] > threshold]


# In[63]:


from scipy import stats
# Outlier Tests
# Dixon's Q Test
outliers_dixon = autinsurance[(np.abs(autinsurance['months_as_customer'] - autinsurance['months_as_customer'].mean()) > threshold * autinsurance['months_as_customer'].std())]
# Print the detected Outliers
outliers_dixon
# Machine Learning Techniques
# Clustering (K-means)


# In[64]:


z_scores = np.abs(stats.zscore(autinsurance['months_as_customer']))
outliers = autinsurance[z_scores > threshold]

# Remove outliers from DataFrame
df_cleaned = autinsurance[z_scores <= threshold]

# Optional: Analyze the modified dataset
print("Original dataset shape:", autinsurance.shape)
print("Cleaned dataset shape:", df_cleaned.shape)


# In[65]:


df_cleaned.to_csv("insurance_claimsV4.csv", index = False)


# In[66]:


df = pd.read_csv("insurance_claimsV4.csv")


# In[67]:


df.shape


# In[68]:


df.columns


# In[69]:


# Rgagde Mohamed Sid Abdalla
def logistic_function(x):
    return 1 / (1 + np.exp(-x))


# In[70]:


# Rgagde Mohamed Sid Abdalla
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


# In[71]:


# Rgagde Mohamed Sid Abdalla
def modele(X, W, b):
    Z = X.dot(W) + b
    A = logistic_function(Z)
    # print(A.shape)
    return A


# In[72]:


def log_loss(y, A):
    return 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))


# In[73]:


def gradients(X, A, y):
    dW = 1/len(y) * np.dot(X.T, A - y)
    db = 1/len(y) * np.sum(A - y)
    return (dW, db)


# In[74]:


def optimisation(X, W, b, A, y, learning_rate):
    dW, db = gradients(X, A, y)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


# In[75]:


def predict(X, W, b):
    A = modele(X, W, b)
    return A >= 0.5


# In[76]:


def regression_logistique(X, y, learning_rate=0.1, n_iter=100):
    # Initialisation
    W, b = initialisation(X)
    loss_history = []
    # Entrainement
    for i in range(n_iter):
        A = modele(X, W, b)
        loss_history.append(log_loss(y, A))
        W, b = optimisation(X, W, b, A, y, learning_rate)
    # Prediction
    plt.plot(loss_history)
    plt.xlabel('n_iteration')
    plt.ylabel('Log_loss')
    plt.title('Evolution des erreurs')
    return W, b


# In[77]:


W, b = regression_logistique(X, y, learning_rate=0.1, n_iter=700)


# In[78]:



y_pred = predict(X, W, b)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)


# In[79]:


pwd

