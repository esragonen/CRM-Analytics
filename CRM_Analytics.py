# CRM ANALYTIC for FLO

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
from astropy.units.quantity_helper.function_helpers import quantile
from xarray.util.generate_ops import inplace

# First Task:
df = pd.read_csv("/Users/esrag/Desktop/MIUUL/Kurs Materyalleri(CRM Analitiği)/FLOMusteriSegmentasyonu/flo_data_20k.csv")

df = df.copy()

df.head()
df.columns
df.describe().T
df.shape
df.isnull().sum()
df.info()

df["total_purchase"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_spending"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()



for i in range(3,7):
    mask = df.columns[i]
    df[mask] = pd.to_datetime(df[mask], format='%Y-%m-%d')

df.info()

df_kanal = df.groupby("order_channel").agg({"master_id": lambda x: x.nunique(),
                                 "total_purchase": lambda x : x.sum(),
                                 "total_spending": lambda x: x.sum()})


df.sort_values("total_spending", ascending=False)[["master_id", "total_spending"]].head(10)

df.sort_values("total_purchase", ascending=False)[["master_id", "total_purchase"]].head(10)

def preparing(data):
    data_copy = data.copy()

    if data.isnull().sum().sum()!=0:
        data.dropna(inplace=True)


    data["total_purchase"] = data["order_num_total_ever_online"] + data["order_num_total_ever_offline"]
    data["total_spending"] = data["customer_value_total_ever_offline"] + data["customer_value_total_ever_online"]

    for i in range(3, 7):
        mask = data.columns[i]
        data[mask] = pd.to_datetime(data[mask], format='%Y-%m-%d')

    return data

# Recency : analiz gününden önce en son ne zaman alışveriş yaptı
# frequency : toplam alışveriş sayısı
# monetary : harcama

df["last_order_date"].max()

today = dt.datetime(year=2021, month=6, day=1)

mask = list(df.columns)
mask[-2:] = ["frequency", "monetary"]
df.columns = mask

df["recency"] = (today - df["last_order_date"])

df.head()

rfm = df[["master_id","recency", "frequency", "monetary"]]
rfm.head()

rfm["recency_sc"] = pd.qcut(rfm["recency"], 5, [5,4,3,2,1])
rfm["frequency_sc"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, [1,2,3,4,5])
rfm["monetary_sc"] = pd.qcut(rfm["monetary"], 5, [1,2,3,4,5])

rfm["rfm"] = rfm["recency_sc"].astype("str") + rfm["frequency_sc"].astype("str")

rfm.head()

seg = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

rfm["segment"] = rfm["rfm"].replace(seg, regex=True)
rfm.head()

 rfm = rfm.drop(rfm.columns[4:8], axis=1)
 rfm.head()


rfm_alter =  rfm.drop("master_id", axis=1)

rfm_alter.groupby("segment").agg(["mean", "min", "max"])

rfm["recency"]=rfm["recency"].dt.days


mask_a = rfm[rfm["segment"]=="champions"]["master_id"]

champ = df[df["master_id"].isin(list(mask_a))]

id_a = champ[champ["interested_in_categories_12"].str.contains("KADIN")]["master_id"]

len(id_a)



rfm["segment"].unique()
lis = [ "cant_loose", "about_to_sleep", "new_customers"]

mask_b=rfm[rfm["segment"].isin(lis)]["master_id"]

erk_coc=df[df["interested_in_categories_12"].str.contains("ERKEK", "COCUK")]

id_b = erk_coc[erk_coc["master_id"].isin(list(mask_b))]["master_id"]

len(id_b)

################################################################################

# Second Task

import pandas as pd
import datetime as dt

df = pd.read_csv("/Users/esrag/Desktop/MIUUL/Kurs Materyalleri(CRM Analitiği)/FLOMusteriSegmentasyonu/flo_data_20k.csv")

mask = df.columns[df.columns.str.contains("date")]
df[mask] = df[mask].apply(pd.to_datetime)
df.info()



def outlier_tresholds(data, var):
    lower = data[var].quantile(0.01)
    upper = data[var].quantile(0.99)
    iqr = upper - lower
    lower_range = lower - 1.5*iqr
    upper_range = upper + 1.5*iqr
    return round(lower_range), round(upper_range)

def replace_with_tresholds(dataf, varib):
    low, up = outlier_tresholds(dataf, varib)
    dataf.loc[(dataf[varib]<low), varib] = low
    dataf.loc[(dataf[varib]>up), varib] = up


replace_with_tresholds(df, "order_num_total_ever_online")
replace_with_tresholds(df,"order_num_total_ever_offline" )
replace_with_tresholds(df,"customer_value_total_ever_offline" )
replace_with_tresholds(df, "customer_value_total_ever_online")


df["total_purch"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_spend"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info()


df["last_order_date"].max()
analiz = dt.datetime(2021, 6, 1)


# recency : ilk ve son alışveriş arasında geçen zaman hafta olarak
df["recency"] = ((df["last_order_date"] - df["first_order_date"]).dt.days)/7
# T : analiz tarihinden ne kadar önce ilk alışveriş yapılmış haftalık
df["T"] = ((analiz - df["first_order_date"]).dt.days)/7

cltv = pd.DataFrame()
cltv["cust_id"] = df["master_id"]
cltv["recency_weekly"] = df["recency"]
cltv["T_weekly"] = df["T"]
cltv["freq"] = df["total_purch"]
cltv["monetary"] = df["total_spend"]/df["total_purch"]
# frequency : tekrar eden toplam satın  alma
cltv = cltv[cltv["freq"]>1]

cltv.head()


from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import  matplotlib.pyplot as plt
bgf =  BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(cltv["freq"],
        cltv["recency_weekly"],
        cltv["T_weekly"])

cltv["pred_3_month"]=bgf.predict(4*3, cltv["freq"], cltv["recency_weekly"], cltv["T_weekly"])

cltv["pred_6_month"]=bgf.predict(4*6, cltv["freq"], cltv["recency_weekly"], cltv["T_weekly"])


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv["freq"], cltv["monetary"])

cltv["exp_average_val"]=ggf.conditional_expected_average_profit(cltv["freq"], cltv["monetary"])

cltv["cltv"]=ggf.customer_lifetime_value(bgf, cltv["freq"], cltv["recency_weekly"], cltv["T_weekly"], cltv["monetary"], time=6, freq="W")

cltv.sort_values("cltv", ascending=False).head(20)

cltv["segment"]=pd.qcut(cltv["cltv"], q=4, labels=["D","C","B", "A"])

cltv.head()
cltv_2 = cltv.drop("cust_id", axis=1)
cltv_2.groupby("segment").agg(["min", "mean", "max"])

##################################################################################

# CRM ANALYTIC for ONLINE RETAIL
# Task 1

data = pd.read_excel("/Users/esrag/Desktop/MIUUL/Kurs Materyalleri(CRM Analitiği)/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df_ = data.copy()

data.describe().T

data.isnull().sum()

data.dropna(inplace=True)

data.isnull().sum()

data["Description"].nunique()

data.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity",ascending=False)

data["Invoice"]=data["Invoice"].astype("str")

data = data[~(data["Invoice"].str.contains("C"))]

data = data[data["Quantity"]>0]

data["total_price"] = data["Quantity"] * data["Price"]

data["InvoiceDate"].max()

today = dt.datetime(2011, 12, 11)

# recency : analiz - son alışveriş
# frequency : toplam işlem sayısı
# monetary : toplam harcadığı

rfm = data.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today - date.max()).days,
                                 "Invoice": lambda x: x.nunique(),
                                 "total_price": lambda z: z.sum()})


rfm.head()

rfm.columns = ["recency", "frequency", "monetary"]
rfm = rfm[(rfm['monetary'] > 0)]

rfm["rec_sc"] = pd.qcut(rfm["recency"], 5, [5,4,3,2,1])
rfm["fre_sc"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, [1,2,3,4,5])
rfm["mon_sc"] = pd.qcut(rfm["monetary"], 5, [1,2,3,4,5])

rfm.head()

rfm["score"] = rfm["rec_sc"].astype("str") + rfm["fre_sc"].astype("str")

seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

rfm["segment"] = rfm["score"].replace(seg_map, regex=True)

rfm.groupby("segment")[["recency", "frequency", "monetary", "score"]].agg(["min", "mean", "max"])

rfm["score"] = rfm["score"].astype("int")

list[rfm.loc[rfm["segment"]=="loyal_customers", :].index]

# Task 2

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

data = data[~data["Invoice"].str.contains("C", na=False)]
data = data[data["Quantity"] > 0]
data = data[data["Price"] > 0]


replace_with_thresholds(data, "Quantity")
replace_with_thresholds(data, "Price")

data["TotalPrice"] = data["Quantity"] * data["Price"]

today_date = dt.datetime(2011, 12, 11)


cltv = data.groupby("Customer ID").agg({
    "InvoiceDate": [lambda x:( x.max() - x.min()).days,lambda x: (today_date - x.min()).days],
    "Invoice" : lambda x: x.nunique(),
    "TotalPrice" : lambda x: x.sum()

})

cltv.head()
cltv.columns=cltv.columns.droplevel(0)

cltv.columns = ['recency', 'T', 'frequency', 'monetary']

cltv["monetary"] = cltv["monetary"] / cltv["frequency"]

cltv = cltv[cltv["frequency"]>1]

cltv["recency"] = cltv["recency"]/7

cltv["T"] = cltv["T"]/7

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv['frequency'],
        cltv['recency'],
        cltv['T'])

cltv["6month"]=bgf.predict(4*6,
        cltv["frequency"],
        cltv["recency"],
        cltv["T"])

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv['frequency'], cltv['monetary'])


cltv1 = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency'],
                                   cltv['T'],
                                   cltv['monetary'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)


cltv12.sort_values(ascending=False)

cltv1.sort_values( ascending=False)

import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)


