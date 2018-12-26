import pandas as pd, datetime as dt

#Original code to read in file, sample it and write it
'df = pd.read_excel("Online Retail.xlsx")'
'df = df.sample(n = 70864)'
'df.to_csv("sample_onret.csv")'

df = pd.read_csv("sample_onret.csv")
df = df.loc[:,'InvoiceNo':]

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
