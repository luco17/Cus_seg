import pandas as pd, datetime as dt, matplotlib as plt

#Original code to read in file, sample it and write it
'df = pd.read_excel("Online Retail.xlsx")'
'df = df.sample(n = 70864)'
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
