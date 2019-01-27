import pandas as pd, datetime as dt, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#Original code to read in file, sample it and write it
'df = pd.read_excel("Online Retail.xlsx")'
'df.to_csv("sample_onret.csv")'

df = pd.read_csv("sample_onret.csv")
df = df.loc[:,'InvoiceNo':]
df.dropna(inplace = True)


#Extracting the date of a given transaction and finding earliest transaction date for a customer
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
def get_day(x): return dt.datetime(x.year, x.month, x.day)
df['InvoiceDay'] = df['InvoiceDate'].apply(get_day)

#Grouping by customer ID to find earliest purchase date
grouping = df.groupby('CustomerID')['InvoiceDay']
df['CohortDay'] = grouping.transform('min')

#Defining a Date function that returns integer date values
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day

# Extracting integer date values for invoices and cohorts
invoice_year, invoice_month, invoice_day = get_date_int(df, 'InvoiceDay')
cohort_year, cohort_month, cohort_day = get_date_int(df, 'CohortDay')

#Creating a datediff index between cohort formation and last invoice
#This calculates recency of users' last purchase
years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month
days_diff = invoice_day - cohort_day

# Assumes 365 days in year, 30 in month
df['CohortIndex'] = years_diff * 365 + months_diff * 30 + days_diff + 1
print(df.head())

df['CohortIndex'].plot(kind = "hist")

###Building cohort tables for calculating retention rates
#Creating cohort and invoice 1st day of month columns
cohort_months_generated = []

for i,j in zip(cohort_year, cohort_month):
    date = dt.datetime(i, j, 1)
    cohort_months_generated.append(date)

df['CohortMonth_clean'] = cohort_months_generated

invoice_months_generated = []

for i,j in zip(invoice_year, invoice_month):
    date = dt.datetime(i, j, 1)
    invoice_months_generated.append(date)

df['InvoiceMonth_clean'] = invoice_months_generated

###Creating a retention table, changing CohortIndex from days since last purchase to months
df['CohortIndex_month'] = round(df['CohortIndex']/30 + 1,0).astype('int64')

grouping = df.groupby(['CohortMonth_clean', 'CohortIndex_month'])

# Count the number of unique month, index combos per customer ID
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique).reset_index()

# Using a pivot to generate the table
cohort_counts = cohort_data.pivot(index = 'CohortMonth_clean', columns = 'CohortIndex_month', values = 'CustomerID')

# Taking the first column as initial size and calculating the % retention as months pass
cohort_sizes = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_sizes, axis = 0)

#Visualizing retention rate
plt.figure(figsize = (8, 6))
plt.title('Cohort retention rate')
sns.heatmap(retention, annot = True, cmap = 'BuGn', fmt = '.0%', vmin = 0.0, vmax = 0.5)
plt.show()

# Analyzing average Unit spend by cohort
grouping = df.groupby(['CohortMonth_clean', 'CohortIndex_month'])
cohort_data = grouping['UnitPrice'].mean().reset_index()

# Creating a pivot
average_quantity = cohort_data.pivot(index = 'CohortMonth_clean', columns = 'CohortIndex_month', values = 'UnitPrice')
print(average_quantity.round(1))

#Visualizing average spend
plt.figure(figsize = (8, 6))
plt.title('Average spend by cohort')
sns.heatmap(average_quantity, annot = True, cmap = 'Blues')
plt.show()

###Developing a Recency, Grequency, Monetary Value model (RFM)
spend_data = {'CustomerID' : np.arange(0,8), 'Spend' : np.random.randint(100, 400, 8), 'Recency_Days' : np.random.randint(30, 400, 8)}
test_df = pd.DataFrame(spend_data)

#Calculating percentiles
spend_quartile = pd.qcut(test_df['Spend'], q = 4, labels = range(1, 5))
test_df['Spend_Quartile'] = spend_quartile
print(test_df.sort_values('Spend'))
'Recency in descending order'
# Store labels from 4 to 1 in a decreasing order
r_labels = list(range(4, 0, -1))
recency_quartiles = pd.qcut(test_df['Recency_Days'], q = 4, labels = r_labels)
test_df['Recency_Quartile'] = recency_quartiles
print(test_df.sort_values('Recency_Days'))

#Using a snapdate for the RFM
df['TotalSum'] = df['UnitPrice'] * df['Quantity']

snap = max(df.InvoiceDate) + dt.timedelta(days = 1)

grouped_df = df.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snap - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

grouped_df.rename(columns = {'InvoiceDate': 'Recency',
                             'InvoiceNo': 'Frequency',
                             'TotalSum': 'MonetaryValue'}, inplace = True)

###Building segments from the RFM analysis
# Creating ascending and descending labels
r_labels = range(3, 0, -1); f_labels = range(1, 4); m_labels = range(1, 4)
r_groups = pd.qcut(grouped_df['Recency'], q = 3, labels = r_labels)
f_groups = pd.qcut(grouped_df['Frequency'], q = 3, labels = f_labels)
m_groups = pd.qcut(grouped_df['MonetaryValue'], q = 3, labels = m_labels)

# Creating columns R, F and M
grouped_df = grouped_df.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)

# Calculate RFM_Score
grouped_df['RFM_Score'] = grouped_df[['R','F','M']].sum(axis = 1)
print(grouped_df['RFM_Score'].head())

# Defining groups of customers
def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Gold'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 9)):
        return 'Silver'
    else:
        return 'Bronze'

# Create the new variable
grouped_df['RFM_Level'] = grouped_df.apply(rfm_level, axis=1)

# Calculate average values for each customer level, and return a size of each segment
rfm_level_agg = grouped_df.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
}).round(1)

# Print the aggregated dataset
print(rfm_level_agg)

###Analysing the RFM distribution
RFM = grouped_df.loc[:,['Recency', 'Frequency', 'MonetaryValue']]

#Broaldy the same mean and standard deviation, opening them up for K means clustering
RFM.describe()
RFM.mean()
RFM.std()

#Visual distribution analysis, none normally distributed, heavy skew
plt.subplot(3, 1, 1); sns.distplot(RFM['Recency'])
plt.subplot(3, 1, 2); sns.distplot(RFM['Frequency'])
plt.subplot(3, 1, 3); sns.distplot(RFM['MonetaryValue'])
plt.show()

#Dealing with skewness, removing negative values from MV
RFM_2 = RFM.loc[(RFM['MonetaryValue'] > 0)]
RFM_log_trans = np.log(RFM_2)

#Normalizing the data to get mean zero and std 1
# Manual
RFM_norm = (RFM - RFM.mean()) / RFM.std()
print(RFM_norm.describe().round(2))

#Running the same operation but using an SKlearn pipeline
scaler = StandardScaler() ;scaler.fit(RFM_log_trans)
RFM_norm = scaler.transform(RFM_log_trans)
#Converting to a PD DF
RFM_norm = pd.DataFrame(RFM_norm, index = RFM_log_trans.index, columns = RFM_log_trans.columns)
RFM_norm.describe().round(2)

#Plotting the new distribution
plt.subplot(3, 1, 1); sns.distplot(RFM_norm['Recency'])
plt.subplot(3, 1, 2); sns.distplot(RFM_norm['Frequency'])
plt.subplot(3, 1, 3); sns.distplot(RFM_norm['MonetaryValue'])
plt.show()

#Running Kmeans with the normalized data
sse = {}
# Fitting KMeans and calculate SSE for each k between 1 and 10
for k in range(1, 11):

    # Initialize KMeans with k clusters and fit it
    kmeans = KMeans(n_clusters = k, random_state = 1).fit(RFM_norm)

    # Assign sum of squared distances to k element of the sse dictionary
    sse[k] = kmeans.inertia_

# Creating an elbow plot to evaluate cluster numbers
plt.title('The Elbow Method'); plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x = list(sse.keys()), y=list(sse.values()))
plt.show()

#Running 3 clusters
kmeans = KMeans(n_clusters = 3, random_state = 1)
kmeans.fit(RFM_norm)
# Extract cluster labels
cluster_labels = kmeans.labels_

# Adding cluster labels
RFM_C = RFM_2.assign(Cluster = cluster_labels)
grouped = RFM_C.groupby(['Cluster'])
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
  }).round(1)
