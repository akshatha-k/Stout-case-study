#!/usr/bin/env python
# coding: utf-8

# In[1]:


import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from heatmap import heatmap, corrplot
sns.set(color_codes=True, font_scale=1.2)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Analysis and Cleaning
# ![data%20cleaning.jpeg](attachment:data%20cleaning.jpeg)

# In[2]:


df = pd.read_csv("Downloads/loans_full_schema.csv")
df.info()


# We drop the columns that contain a lot of missing values (more than 75% missing values). While "months_since_last_delinq" column also has significant missing values (5658/1000 missing), we decide to keep this in as we suspect that lenders might weigh this highly.

# In[3]:


df.drop(["annual_income_joint", "verification_income_joint", "debt_to_income_joint", "months_since_90d_late"], axis=1, inplace= True)


# In[7]:


df_states = df.groupby('state', sort=False)["interest_rate"].mean().reset_index(name ='interest_rate')
# for col in df.columns:
#     df[col] = df[col].astype(str)
df_states['text'] = df_states['state'] + '<br>' + 'Interest Rate' + df_states['interest_rate'].astype(str)

fig = go.Figure(data=go.Choropleth(
    locations=df_states['state'],
    z=df_states['interest_rate'],
    locationmode='USA-states',
    colorscale='Blues',
    autocolorscale=False,
    text=df_states['text'], # hover text
    marker_line_color='darkgray', # line markers between states
    colorbar_title="Percentage"
))

fig.update_layout(
    title_text='Average Interest Rate by State<br> (Hover over state to see individual rates)',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()


# Let's now view some basic statistics about the the numeric columns.

# In[8]:


df.describe()


# In[9]:


df.drop(['paid_total','paid_principal','paid_interest',
         'paid_late_fees','grade', 'sub_grade'] ,1, inplace=True)


# We don't see any obvious logical discrepancies in the data- any weird values etc. So we keep the numerical data as is.

# In[10]:


plt.figure(figsize=(8, 8))
plt.title('Interest rate is NOT highly correlated with any single variable')
corrplot(df.corr(), size_scale=300);


# We make the above correlation heatmap that uses more than just colors to show correlations- making it easier to read, especially in the presence of a large number of variables. For a start, we do not see any single variable that the interest rate is directly highly correlated with.

# ### Employee Title
# emp_title might be a free text field on the application form or a list of currated employment titles. Let’s examine how many unique values exist:

# In[11]:


print(df.emp_title.value_counts().head())
print(df.emp_title.value_counts().tail())
df.emp_title.unique().shape


# Taking a look at the head vs tail of the doc shows some specific titles such as full time rn (registered nurse) and sr loan officers. Further there are 4742 unique values in 10000 entries. I feel comfortable assessing that this data won’t be meaningful and any relationship we might observe might be due to confounding relationships. A more advanced implementation might look to group all these job descriptions into categories and/or examine if Lending Club’s model looks at (income + job) versus just income.
# 
# To explain confounding relationships mentioned above, let us assume that class A is taught by teacher A who uses a certain mathematics app A to aid her lessons. Whereas class B is taught by teacher B using app B. Now, if a class performs better on test scores then here is no way to determine if differences in scores between the two classes were caused by either or both of the independent variables (teacher effectiveness and app effectiveness).
# 
# Applying this example to our dataset, Registered Nurses, or RNs, who have higher education requirements and receive above average pay, might be granted A grade loans on average. Is this due to them working as RNs or having a regular source of income or having a 4-year degree or their salary? Would we see the same effect for Physician Assistants who go to school for longer and receive even more pay? What about Certified Nursing Assistants (CNAs) who are on the oppposite spectrum?
# ![confounding.jpeg](attachment:confounding.jpeg)

#  ### Employment Length
# Leaving this variable in might contradict our decision to drop the employment tile since it conveys socio-economic seniority. A Computer Scientist 5 years into their career would generally have a larger salary than a Kindergarden teacher 10 years into their career. Arguably it might be powerful to combine a grouped, matched, and reduced set of employment titles with their length to create a “purchasing power” metric. However, since employment length is an easy scalar, let’s leave it in for now.
# 
# 

# ### Visualizing income rates with probability of loan payment

# In[15]:


def fully_paid(loan_status):
    if (loan_status == 'Fully Paid')| (loan_status=='Does not meet the credit policy. Status:Fully Paid'):
        return(1)
    elif (loan_status == 'Charged Off') | (loan_status=='Does not meet the credit policy. Status:Charged Off'):
        return(0)
    else:
        return(-1)
df['Paid'] = df['loan_status'].apply(fully_paid)
df_paid = df[df['Paid']!=-1]

print('Maximum annual income',max(df_paid['annual_income']))
print('Minimum annual income',min(df_paid['annual_income']))
print('Mean annual income',df_paid['annual_income'].mean())


# In[16]:


def income_class(income):
    if income<50000:
        return 'low'
    if (income>50000) and (income<120000):
        return 'medium'
    if income>120000:
        return 'high'
df['income_class'] = df['annual_income'].apply(income_class)

f, ax = plt.subplots(1,2, figsize=(16,8))

labels = ['Medium Income','Low Income', 'High Income']
df["income_class"].value_counts().plot.pie(autopct='%1.2f%%', ax=ax[0], shadow=True, 
                                           labels=labels, fontsize=12, startangle=70)


sns.barplot(x="income_class", y="Paid", data=df_paid)
ax[1].set(ylabel="Probability of payment")
f.show()


# The annual income of borrowers goes from 0 to 2300000 dollars, showing the diverse profile of borrowers. The majority of borrowers have an annual income around 86000 dollars.
# 
# 30% of borrowers have low income (Annual income < 50000 dollars), 56% medium income (120000 dollars > Annual income> 50000 dollars) and 14% high income (Annual income > 120000 dollars).
# 
# The probability of payment appears independent of income classification.

# In[17]:


df.drop(["Paid", "income_class"], axis=1, inplace= True)


# ### Loan Status
# Loan status is mutable value that represents the current state of the loan. If anything we might want to examine if all the independent variables and/or interst rate to determine the probability of the loan status at some time. In this work we want to predict the interest rate granted to an applicant at loan creation. Thus we do not care about this variable and will drop it after examining it.
# 
# When examining the distribution among the loan statuses, October has the highest amount of loans right ahead of the holiday season. To clean up the space let’s replace the column value with a month only and convert the column back to a string or object type.

# In[18]:


print(df.loan_status.value_counts())
df.drop(['loan_status'],1, inplace=True)


# ### Issue Month and other string variables

# In[19]:


df.issue_month.value_counts().sort_index().plot(kind='bar')


# ### Post Loan Attributes

# In[21]:


print(df.homeownership.value_counts())
print(df.loan_purpose.value_counts())
print(df.application_type.value_counts())
print(df.initial_listing_status.value_counts())
print(df.disbursement_method.value_counts())


# In[23]:


for col in ['issue_month', 'state', 'homeownership', 'loan_purpose', 'application_type','initial_listing_status','disbursement_method']:
    df[col] = df[col].astype('category')


# In[29]:


features = ['emp_length', 'state', 'homeownership', 'annual_income', 'debt_to_income', 'delinq_2y',
       'months_since_last_delinq', 'earliest_credit_line',
       'inquiries_last_12m', 'total_credit_lines', 'open_credit_lines',
       'total_credit_limit', 'total_credit_utilized',
       'num_collections_last_12m', 'num_historical_failed_to_pay',
       'current_accounts_delinq', 'total_collection_amount_ever',
       'current_installment_accounts', 'accounts_opened_24m',
       'months_since_last_credit_inquiry', 'num_satisfactory_accounts',
       'num_accounts_120d_past_due', 'num_accounts_30d_past_due',
       'num_active_debit_accounts', 'total_debit_limit',
       'num_total_cc_accounts', 'num_open_cc_accounts',
       'num_cc_carrying_balance', 'num_mort_accounts',
       'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt',
       'loan_purpose', 'application_type', 'loan_amount', 'term', 'installment', 
       'disbursement_method', 'balance']
target = 'interest_rate' 

final_df = df[features + [target]]


# In[32]:


categorical_variables = []
for feat_name, feat_type in zip(final_df.columns, final_df.dtypes):
    if feat_type.name == 'category':
        categorical_variables.append(feat_name)
        
for feature in categorical_variables:
    
    final_df_one_hot_encoded = pd.get_dummies(final_df[feature],prefix=feature)
    #print loans_one_hot_encoded
    
    final_df = final_df.drop(feature, axis=1)
    for col in final_df_one_hot_encoded.columns:
        final_df[col] = final_df_one_hot_encoded[col]
    
print(final_df.head(2))        
print(final_df.columns)


# In[43]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# ## Training and Prediction
# ![training.jpeg](attachment:training.jpeg)

# ### Simple - Linear Regression

# In[59]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

y = final_df.interest_rate
X_train, X_test, y_train, y_test = train_test_split(final_df, y, test_size=0.2)


# In[58]:


lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print("Score {}".format(model.score(X_test, y_test)))


# ### Gradient boosted Regression Trees

# We won’t spend too much time in this tuning, but in general GBTs give us threes knobs we can tune for overfitting: (1) Tree Structure, (2) Shrinkage, (3) Stochastic Gradient Boosting. In the interest of time we’ll do a simple grid search amongst a hand chosen set of hyper-parameters.
# 
# One of the most effective paramerters to tune for when working with a large feature set is max_features as it introduces a notion of randomization similar to Random Forests. Playing with max features allows us to perform subsampling of our feature space before finding the best split node. A max_features setting of .20 for example would grow each tree on 20% of the featureset. Conversely the subsample feature would use 20% of the training data (all features). Subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.

# In[82]:


from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


# In[83]:


from sklearn.model_selection import GridSearchCV
param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              'max_features': [1.0, 0.3, 0.1]
              }
# param_grid = {'learning_rate': [0.1],
#               'max_depth': [4],
#               'min_samples_leaf': [3],
#               'max_features': [1.0],
#               }

est = GridSearchCV(ensemble.GradientBoostingRegressor(n_estimators=100),
                   param_grid, n_jobs=4, refit=True)

est.fit(X_train, y_train)

best_params = est.best_params_


# In[84]:


get_ipython().run_cell_magic('time', '', 'est = ensemble.GradientBoostingRegressor(n_estimators=2000).fit(X_train, y_train)')


# In[85]:


est.score(X_test,y_test)


# In[86]:


sns.set(font_scale=1, rc={"lines.linewidth":1.2}) 


Iterations = 2000
# compute test set deviance
test_score = np.zeros((Iterations,), dtype=np.float64)

for i, y_pred in enumerate(est.staged_predict(X_test)):
    test_score[i] = est.loss_(y_test, y_pred)

plt.figure(figsize=(14, 6)).subplots_adjust(wspace=.3)

plt.subplot(1, 2, 1)
plt.title('Deviance over iterations')
plt.plot(np.arange(Iterations) + 1, est.train_score_, 'dodgerblue',
         label='Training Set Deviance', alpha=.6)
plt.plot(np.arange(Iterations) + 1, test_score, 'firebrick',
         label='Test Set Deviance', alpha=.6)
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


plt.subplot(1, 2, 2,)
# Top Ten
feature_importance = est.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10), np.array(df.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')


# There is some bug in the gradient boost- given more time, I would fix the error.
# 
# I also understand that the training and prediction part has a LOT of scope for improvement as I hurried through it- so adding in more visualizations to make it easy for the layperson audience would be a top priority. 
# 
# I would also try to perform dimensionality reduction so we are not dealing with 44 columns, but only the ones with higher importance. 
# Given that the size of the dataset is super small, I would NOT use any neural network methods- however, if the dataset size were to increase- then it may be a feasible option.

# In[ ]:




