from operator import eq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sys
import sklearn
import io
import random

df_master = pd.DataFrame()
df_chi = pd.DataFrame()
df_pca = pd.DataFrame()
df_rfi = pd.DataFrame()
selected_feat = 1

#data exploration
def read_dataset():
    global df_master
    df_anomaly_master = pd.read_csv("https://www.kaggle.com/datasets/vickyg0609/mcafyp/CIC_IDS_2018.csv", low_memory=False)
    df_benign_master = pd.read_csv("https://www.kaggle.com/datasets/vickyg0609/mcafyp/benign.csv", low_memory=False)
    df_master = pd.concat([df_anomaly_master, df_benign_master])

def get_data_shape():
    global df_master
    return df_master.shape

def get_data_head():
    global df_master
    return df_master.head()

def get_data_tail():
    global df_master
    return df_master.tail()

def get_label_counts():
    global df_master
    c = df_master.Label.value_counts().to_frame()
    c.rename(columns = { "Label" : "count"}, inplace = True)
    c.index.names = ["Attacks"]
    c.reset_index(level=0, inplace=True)
    return c

#pre-processing....
def drop_timestamp():
    global df_master
    df_master = df_master.drop(["Timestamp"], axis="columns")
    return True

def datatype_casting():
    global df_master
    for col_name in df_master.columns:
        if col_name != "Label":
            df_master[col_name] = pd.to_numeric(df_master[col_name],errors = 'coerce')  
    df_master = df_master.convert_dtypes()
    return True

def data_cleaning():
    global df_master
    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_master = df_master.dropna()
    df_master.isnull().any()
    return True

def data_describe():
    global df_master
    return df_master.describe()


#feature selection

def chi(n_comp: int):                                                        #Chi-squared test
    global df_master, df_chi
    dfn = df_master.copy()
    X = dfn.iloc[:, :-1]
    y = dfn.iloc[:, -1]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    scaler.fit(X)
    X_scaled = scaler.transform(X)

    from sklearn.feature_selection import SelectKBest, chi2

    bestfeatures = SelectKBest(score_func=chi2, k = 78)
    fit = bestfeatures.fit(X_scaled, y)
    dfscores = pd.DataFrame(fit.scores_).convert_dtypes()
    dfcolumns = pd.DataFrame(X.columns)
    featuresScores = pd.concat([dfcolumns, dfscores], axis = 1)
    featuresScores.columns = ["attributes", "score"]
    attr = featuresScores.nlargest(80, "score")
    cols = attr.iloc[:n_comp, 0].values
    df_chi = dfn[cols]   
    return featuresScores.nlargest(n_comp, "score")

def pca(n_variance: float):                                                        #principal_component_analysis
    global df_master, df_pca
    dfn = df_master.copy()
    X = dfn.iloc[:, :-1]
    y = dfn.iloc[:, -1]
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()

    scalar.fit(X)
    X_scaled_pca = scalar.transform(X)
    from sklearn.decomposition import PCA
    pca = PCA(n_variance)

    pca.fit(X_scaled_pca)
    X_pca = pca.transform(X_scaled_pca)

    plt.figure(figsize=(10,6))
    plt.scatter(x=[i+1 for i in range(len(pca.explained_variance_ratio_))],
                y=pca.explained_variance_ratio_,
            s=200, alpha=0.75,c='orange',edgecolor='k')
    plt.grid(True)
    plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=18)
    plt.xlabel("Principal components",fontsize=12)
    plt.xticks([i+1 for i in range(len(pca.explained_variance_ratio_))],fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Explained variance ratio",fontsize=12)
    plt.savefig("static/images/pca.png")

    return pca.explained_variance_ratio_


def rfi(n_estimator: int):
    global df_master, df_rfi, selected_feat
    if not df_rfi.empty:
        return selected_feat
    dfn = df_master.copy()
    X = dfn.iloc[:, :-1]
    y = dfn.iloc[:, -1]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    sel = SelectFromModel(RandomForestClassifier(n_estimators = n_estimator))
    sel.fit(X, y)
    selected_feat= X.columns[(sel.get_support())]
    df_rfi = dfn[selected_feat]

    return selected_feat



def classification(c_type, feat_sel, criterion):
    global df_chi, df_pca, df_rfi, df_master
    dfn = df_master.copy()
    if c_type == "binary":
        dfn["Label"] = np.where(dfn["Label"] == "Benign", 0, 1)
    X = dfn.iloc[:, :-1]
    y = dfn.iloc[:, -1]
    if feat_sel == "chi":
        from sklearn.preprocessing import StandardScaler
        scalar = StandardScaler()
        scalar.fit(df_chi)
        X_scaled_chi = scalar.transform(df_chi)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_chi, y, test_size=0.3)

        from sklearn.tree import DecisionTreeClassifier
        clf_gini = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0)
        clf_gini.fit(X_train, y_train)

        y_pred_gini = clf_gini.predict(X_test)
        from sklearn.metrics import accuracy_score
        test_acc = accuracy_score(y_test, y_pred_gini)              #test accuracy
        test_acc = test_acc * 100
        test_acc = round(test_acc, 2)
        y_pred_train_gini = clf_gini.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train_gini)      #train accuracy
        train_acc = train_acc * 100
        train_acc = round(train_acc, 2)

        return (train_acc, test_acc)
        


'''
n_arow, n_acol = df_anomaly_master.shape
n_nrow, n_ncol = df_benign_master.shape
print("Number of rows(Anamoly)  :", n_arow)
print("Number of rows(Normal)   :", n_nrow)
print("Number of columns        :", n_ncol)


n_row, n_col = df_master.shape
print("**************Records in Dataset************")
print("Number of rows     :", n_row)
print("Number of columns  :", n_col)


# <br/>

# In[7]:


df_master.head()


# In[8]:


df_master.tail()


# <br/>

# Removing/droping the timestamp column

# In[9]:


df_master = df_master.drop(["Timestamp"], axis="columns")


# <br/>

# In[10]:


# converting the column values to their dtypes
for col_name in df_master.columns:
    if col_name != "Label":
        df_master[col_name] = pd.to_numeric(df_master[col_name],errors = 'coerce')  
df_master = df_master.convert_dtypes()


# In[11]:


df_master.info()


# <br/> 

#  
# #### Data cleaning
# Removing the null values

# In[12]:


df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
df_master = df_master.dropna()
df_master.isnull().any()


# In[13]:


df_master.describe()


# <br/>

# In[14]:


print("*******Attack types and it counts in dataset***********")
df_master.Label.value_counts()


# <br/>

# In[15]:


dfn = df_master.copy()


# Encoding Label *Anamoly* as 1 and the *Benign* as 0

# In[16]:


dfn["Label"] = np.where(dfn["Label"] == "Benign", 0, 1)


# <br/>

# In[17]:


X = dfn.iloc[:, :-1]
y = dfn.iloc[:, -1]


# ## Feature Selection

# ## A. Filter methods
# Filter methods pick up the intrinsic properties of the features measured via univariate statistics instead of cross-validation performance. These methods are faster and less computationally expensive than wrapper methods. When dealing with high-dimensional data, it is computationally cheaper to use filter methods.

# ### A.1 Chi Square Test
# A chi-square test is a statistical test used to compare observed results with expected results. The purpose of this test is to determine if a difference between observed data and expected data is due to chance, or if it is due to a relationship between the variables you are studying.

# ### Feature Scaling
# MinMaxScaler

# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X)
X_scaled = scaler.transform(X)


# #### train_test_split
# train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data. 

# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# In[67]:


from sklearn.feature_selection import SelectKBest, chi2

bestfeatures = SelectKBest(score_func=chi2, k = 78)
fit = bestfeatures.fit(X_scaled, y)


# In[68]:


dfscores = pd.DataFrame(fit.scores_).convert_dtypes()
dfcolumns = pd.DataFrame(X.columns)


# In[69]:


featuresScores = pd.concat([dfcolumns, dfscores], axis = 1)
featuresScores.columns = ["attributes", "score"]


# In[70]:


featuresScores.nlargest(80, "score")


# In[118]:


attr = featuresScores.nlargest(80, "score")
cols = attr.iloc[:21, 0].values                 # getting the top 22 columns
df_chi = dfn[cols]                              # creating dataframe


# In[ ]:





# In[183]:


dec_tree(df_chi, y)


# In[72]:


cols


# In[73]:


df_chi.head()


# In[77]:


df_chi.shape


# In[78]:


y.shape


# <br/>

# ### Principal Component Analysis
# 
# Standard Scaler

# In[120]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scalar.fit(X)
X_scaled_pca = scalar.transform(X)


# <br/>

# In[174]:


from sklearn.decomposition import PCA
pca = PCA(0.99)

pca.fit(X_scaled_pca)
X_pca = pca.transform(X_scaled_pca)


# In[175]:


att_cnt = X_pca.shape[1]
print("Reduced to", att_cnt, "attributes")


# #### Plotting the explained variance ratio

# In[176]:


pca.explained_variance_ratio_


# <br/>

# In[177]:


plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=18)
plt.xlabel("Principal components",fontsize=12)
plt.xticks([i+1 for i in range(len(pca.explained_variance_ratio_))],fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Explained variance ratio",fontsize=12)
plt.show()


# In[187]:


dec_tree(X_pca, y)


# <br/>

# ### Random Forest Importance
# Random Forests is a kind of a Bagging Algorithm that aggregates a specified number of decision trees. The tree-based strategies used by random forests naturally rank by how well they improve the purity of the node, or in other words a decrease in the impurity (Gini impurity) over all trees. Nodes with the greatest decrease in impurity happen at the start of the trees, while notes with the least decrease in impurity occur at the end of trees. Thus, by pruning trees below a particular node, we can create a subset of the most important features.

# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[32]:


X_train.shape


# In[163]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 21))
sel.fit(X, y)


# In[164]:


sel.get_support()


# In[165]:


selected_feat= X.columns[(sel.get_support())]
len(selected_feat)


# In[166]:


print(selected_feat)


# In[167]:


df_rfi = dfn[selected_feat]   


# In[186]:


dec_tree(df_rfi, y)


# <br/>

# <br/>

# # Decison Tree Classification
# Decision tree is a non-parametric supervised learning technique, it is a tree of multiple decision rules, all these rules will be derived from the data features.

# In[85]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scalar.fit(df_chi)
X_scaled_chi = scalar.transform(df_chi)


# In[86]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled_chi, y, test_size=0.2)


# <br/>

# In[87]:


from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
# fit the model
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
print("\n\n")


# In[88]:


y_pred_train_gini = clf_gini.predict(X_train)
print("y_pred_train_gini:",y_pred_train_gini)
print("\n\n")
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
print("\n\n")
# print the scores on training and test set


# In[83]:


print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
print("\n\n")
print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
print("\n\n")


# In[43]:



clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)


# In[44]:


y_pred_en = clf_en.predict(X_test)
print("Y_PRED_EN:",y_pred_en)
print("\n\n")
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

print("\n\n")


# In[45]:


y_pred_train_en = clf_en.predict(X_train)
print("y_pred_train_en:",y_pred_train_en)
print("\n\n")
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))
print("\n\n")
# print the scores on training and test set


# In[46]:


print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))
print("\n\n")
print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))
print("\n\n")


# In[47]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', cm)
print("\n\n")
from sklearn.metrics import classification_report
print("\n\n")
print("CLASSIFICATION REPORT:\n",classification_report(y_test, y_pred_en))


# In[48]:


from sklearn import tree
tree.plot_tree(clf_gini)


# In[49]:


tree.plot_tree(clf_en)


# <br/>

# In[113]:


def dec_tree(X, y):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()

    scalar.fit(X)
    X_scaled_chi = scalar.transform(X)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.tree import DecisionTreeClassifier
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    # fit the model
    clf_gini.fit(X_train, y_train)
    y_pred_gini = clf_gini.predict(X_test)
    from sklearn.metrics import accuracy_score

    print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
    print("\n\n")
    y_pred_train_gini = clf_gini.predict(X_train)
    print("y_pred_train_gini:",y_pred_train_gini)
    print("\n\n")
    print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
    print("\n\n")
    # print the scores on training and test set
    print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
    print("\n\n")
    print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
    print("\n\n")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_en)

    print('Confusion matrix\n\n', cm)
    print("\n\n")
    from sklearn.metrics import classification_report
    print("\n\n")
    print("CLASSIFICATION REPORT:\n",classification_report(y_test, y_pred_en))

'''