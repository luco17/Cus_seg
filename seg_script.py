import pandas as pd, datetime as dt, matplotlib.pyplot as plt, seaborn as sns, numpy as np

#Original code to read in file, sample it and write it
'df = pd.read_excel("Online Retail.xlsx")'
'df.to_csv("sample_onret.csv")'

df = pd.read_csv("sample_onret.csv")
df = df.loc[:,'InvoiceNo':]
df.dropna(inplace = True)

#Extracting the date of a given transaction and finding earliest transaction date for a customer
def get_day(x): return dt.datetime(x.year, x.month, x.day)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
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

###Creating a retention table
df['CohortIndex_month'] = round(df['CohortIndex']/30 + 1,0).astype('int64')

grouping = df.groupby(['CohortMonth_clean', 'CohortIndex_month'])

# Count the number of unique month, index combos per per customer ID
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
