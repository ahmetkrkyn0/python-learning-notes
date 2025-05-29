################################################################
### PYTHON İLE VERİ ANALİZİ ( DATA ANALYSIS WITH PYTHON )
################################################################
# - NumPy
# - Matplotlib
# - Pandas
# - Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

#################
# NUMPY
#################

# Why Numpy?
# Creating Numpy Arrays
# Attibutes of Numpy Arrays
# Reshaping (Yeniden Şekillendirme)
# Index Selection
# Slicing
# Fancy Index
# Conditions on Numpy
# Mathematical Operations

import numpy as np
a = [1,2,3,4]
b = [2,3,4,5]

ab = []

for i in range(0,len(a)):
    ab.append(a[i]*b[i])

a = np.array(a)
b = np.array(b)
a * b


# Creating Numpy Arrays
import numpy as np
np.array([1,2,3,4,5])
type(np.array([1,2,3,4,5]))

np.zeros(10, dtype = int) # 10 tane int olarak 0 oluştur
np.random.randint(0, 10, size = 10) # 0 ile 10 arasında 10 tana rastgele int seç
np.random.normal(10, 4, (3,4))

# Attibutes of Numpy Arrays

import numpy as np
#ndim : boyut sayısı
#shape : boyut bilgisi
#size : toplam eleman sayısı
#dtype : array veri tipi

a = np.random.randint(0,10, size = 5)
a.ndim
a.shape
a.size
a.dtype


# Reshaping (Yeniden Şekillendirme)
import numpy as np

np.random.randint(1, 10, size = 9)
np.random.randint(1, 10, size = 9).reshape(3,3)
ar = np.random.randint(1, 10, size = 9)
ar.reshape(3,3)


# Index Selection
import numpy as np
a = np.random.randint(10, size = 10)
a[0] #np.int32(3)
a[0:3] #array([3, 1, 3]
a[0] = 999

m = np.random.randint(10, size = (3,5))
m[0,0]
m[1,1]
m[2,3] = 999

m[2,3] = 10.2
m[:, 0] # ilk olarak tüm satırları seç ve 0.indextekileri çağır
m[0, :] # 0.satırın tüm sütünlarını getir
m[0:2, 0:3] #satırlarda 0dan 2ye kadar al, sütunlarda 0dan 3 e kadar al


# Fancy Index
import numpy as np
v = np.arange(0,30,3) # 0dan 30a kadar 3er 3er artacak şekilde array oluştur
v[1]
v[4]
catch = [1,4,0]
v[catch] # catch listesindekileri index olarak kullanıp o indexteki sayıları yazdırır


# Conditions on Numpy
import numpy as np
v = np.array([1,2,3,4,5,6,7,8,9,10])

## Klasik döngü ile
ab = []
for i in v:
    if i < 3:
        ab.append(i)

## Numpy ile
v[v < 3]


# Mathematical Operations
import numpy as np
v = np.array([1,2,3,4,5,6,7,8,9,10])
v / 5
v*10
v + 10
v - 10
v ** 2
v % 2

# buradaki işlemlerin hiçbiri kalıcı değildir, eğer kalıcı olmasını isterseniz bir değişkene atayarak işlem yapabilirsiniz.
np.subtract(v,1) #arrayden her bir elemandan 1 sayısını çıkar
np.add(v,1) #arraydeki her bir elemanı 1 ekler
np.mean(v)
np.sum(v)
np.max(v)
np.min(v)
np.std(v)
np.median(v)
np.var(v)
np.sqrt(v)
np.cumprod(v)
np.cumsum(v)
np.log(v)

dir(np) #numpy için kullanılabilecek fonksiyonları listeler

# NumPy ile iki bilinmeyenli denklem çözümü

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10
a = np.array([[5,1], [1,3]])
b = np.array([12,10])
np.linalg.solve(a,b)

#numpy sabit tipte veri saklar ve çeşitli kolaylıklar sağlar.
#numpy ile veri setleri üzerinde işlem yapabiliriz. Genellikle bu tercih edilir.

arr = np.array([1,2,3,4,5,6,7])
arr[-3:-1]


#################
# PANDAS
#################
# veri analizi ya da veri manipülasyonu için kullanılan bir paket.

# Pandas Series
# Reading Data
# Quick Look at Data
# Selection in Pandas
# Aggregation & Grouping
# Apply and Lambda
# Join Operations


# Pandas Series #
import pandas as pd
s = pd.Series([1,2,3,4,5])
del s

s = pd.Series([1,2,3,4,5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values #indexlerle ilgilenmediğimiz için burada bunu numpy arraya döndürdü
type(s.values) #numpy.ndarray
s.head(3) # baştan 3 eleman al
s.tail(3) # sondan 3 eleman al


# Reading Data
import pandas as pd
df = pd.read_csv("data/Advertising.csv")
df.head()

# Pandas cheatsheet
df.max()
df.min()


# Quick Look at Data #
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any() # her bir sütun için NULL olup olmadığını kontrol eder.
df.isnull().sum() # her bir sütunda kaç tane NULL olduğunu gösterir.
df["sex"].head()
df["sex"].value_counts() # cinsiyetlerin sayısı yani kaç kadın kaç erkek olduğunu gösterir.



# Selection in Pandas #
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13] #0dan 13e kadar
df.drop(0, axis = 0).head()

delete_indexes = [1,3,5,7]
df.drop(delete_indexes, axis = 0).head(10)

# kalıcı hale getirmek için
# df = df.drop(delete_indexes, axis = 0)
# df.drop(delete_indexes, axis = 0, inplace = True)

# Değişkeni indexe Çevirmek #

df["age"].head()
df.age.head()
df.index = df["age"]

df.drop("age", axis = 1).head()

df.drop("age", axis = 1, inplace=True)
df.head()

# İndexi değişkene çevirmek #

df.index
df["age"] = df.index
df.head()
df.drop("age", axis = 1, inplace=True)

df.reset_index().head()
df = df.reset_index()
df.head()

# Değişkenler Üzerinde İşlemler #
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

#dataframe'in içinde var mı yok mu kontrol
"age" in df
df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head()) #pandas.core.series.Series
type(df[["age"]].head()) #pandas.core.frame.DataFrame

df[["age", "alive"]]

col_names = ["age","adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"]*100 / df["age2"]

df.drop("age3", axis = 1).head()

df.drop(col_names, axis=1).head()

df.loc[:, ~df.columns.str.contains("age")].head()


# iloc & loc #
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

#iloc: integer based selection
df.iloc[0:3]
df.iloc[0,0]

#loc: label-based selection
df.loc[0:3]
df.loc[0, "age"]
df.loc[0:3, ["age", "alive"]]
df.iloc[0:3, 0:3]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]


# Koşullu Seçim (Conditional Selection)
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count() #yaşı 50den büyük olan kaç kişi var

df.loc[df["age"] > 50,["age", "class"]].head()

#aynı anda iki koşul varsa ayrı ayrı parantez içine alınır#
df.loc[(df["age"] > 50) & (df["sex"] == "male"),["sex", "age", "class"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["sex", "age", "class", "embark_town"]]

df_new["embark_town"].value_counts()



### Aggregation & Grouping ###

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - sum()
# - std()
# - var()
# - quantile()

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town": "count",
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean", "sum"], "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": "mean",
    "survived": "mean",
    "sex": "count"})


# Pivot Table #
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", ["embarked","class"])

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.head()

df.pivot_table("survived", "sex", ["new_age", "class"], observed = True)

pd.set_option('display.width', 500)


# Apply and Lambda #
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
      df[col] = df[col]/10
df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")].head()



# Join işlemleri #
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2], ignore_index=True)


# Merge ile birleştirme işlemleri #

df1 = pd.DataFrame({'employees': ['John', 'Mary', 'Lisa'],
                    'group': ['A', 'B', 'A']})
df2 = pd.DataFrame({'employees': ['John', 'Mary'],
                    'start_date': [2008, 2012]})

df3 = pd.merge(df1, df2)

#Quiz
import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
subset = df.loc[0:1, ['A', 'B']]


import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
subset = df.iloc[0:2, [0, 1]]

