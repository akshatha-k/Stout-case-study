#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[34]:


df = pd.read_csv("Downloads/casestudy.csv")
df.set_index('customer_email')
df.info()


# # Total revenue for current year

# In[19]:


df.groupby(['year']).sum()


# # New Customer Revenue 

# In[20]:


cust_2015 = df[df['year']==2015]
cust_2015.set_index('customer_email')
cust_2016 = df[df['year']==2016]
cust_2016.set_index('customer_email')
cust_2017 = df[df['year']==2017]
cust_2017.set_index('customer_email')

key_diff_16 = set(cust_2016.customer_email).difference(cust_2015.customer_email)
where_diff_16 = cust_2016.customer_email.isin(key_diff_16)

print("New revenue 2016: {}".format(sum(cust_2016[where_diff_16].net_revenue)))

key_diff_17 = set(cust_2017.customer_email).difference(cust_2016.customer_email)
where_diff_17 = cust_2017.customer_email.isin(key_diff_17)

print("New revenue 2017: {}".format(sum(cust_2017[where_diff_17].net_revenue)))


# # Existing Customer Growth

# In[29]:


cust_2015_16 = cust_2015.merge(cust_2016, how='left')
print("Existing customer growth 2016: {}".format(sum(cust_2015_16[cust_2015_16['year']==2016].net_revenue)- sum(cust_2015_16[cust_2015_16['year']==2015].net_revenue)))

cust_2016_17 = cust_2016.merge(cust_2017, how='left')
print("Existing customer growth 2016: {}".format(sum(cust_2016_17[cust_2016_17['year']==2017].net_revenue)- sum(cust_2016_17[cust_2016_17['year']==2016].net_revenue)))


# # New customers

# In[35]:


print("New customers 2016")
print(cust_2016[where_diff_16])


# In[31]:


print("New customers 2017")
print(cust_2016[where_diff_16])


# In[ ]:




