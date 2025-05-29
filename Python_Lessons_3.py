########################################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ ( ADVANCED FUNCTIONAL EDA )
########################################################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


####################################
# 1. Genel Resim
####################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
    print("################# SHAPE ###################")
    print(dataframe.shape)
    print("################# TYPES ###################")
    print(dataframe.dtypes)
    print("################# INFO ###################")
    print(dataframe.info())
    print("################# HEAD ###################")
    print(dataframe.head(head))
    print("################# TAIL ###################")
    print(dataframe.tail(head))
    print("################# NULL ###################")
    print(dataframe.isnull().sum())
    print("################# DESC ###################")
    print(dataframe.describe().T)

check_df(df)

df = sns.load_dataset("tips")
check_df(df)


########################################################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
########################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
sns.countplot(x='parch', data= df, palette='Set2')
plt.show()
df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()

#*Önemli*#
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
#*Önemli*#
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
#*Önemli*#
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
#*Önemli*#
cat_cols = cat_cols + num_but_cat
#*Önemli*#
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()
[col for col in df.columns if col not in cat_cols]

len(df)
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################################")
cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, hue=col_name, legend=False, palette='Set2')
        plt.show(block=True)

cat_summary(df,"sex", plot = True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

plt.clf()
plt.close('all')


df["adult_male"].astype(int)


########################################################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
########################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numeric_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numeric_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    print(f"######################{col}######################")
    num_summary(df, col)
    print("################################################")



# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi #
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

#docstring
def grap_col_names(dataframe, cat_th=10, car_th=20):
    """
    veri setindeki kategorik, numerik ve kategorik fakat kardinal olan değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
    car_th: int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri.
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi.
    num_cols: list
        Numerik değişken listesi.
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi.
    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı.
    num_but_cat cat_cols'un içerisinde olması gerekmektedir.
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grap_col_names(df)



#########################################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#########################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")


def grap_col_names(dataframe, cat_th=10, car_th=20):
    """
    veri setindeki kategorik, numerik ve kategorik fakat kardinal olan değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
    car_th: int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri.
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi.
    num_cols: list
        Numerik değişken listesi.
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi.
    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı.
    num_but_cat cat_cols'un içerisinde olması gerekmektedir.
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, hue=col_name, legend=False, palette='Set2')
        plt.show(block=True)

cat_cols, num_cols, cat_but_car = grap_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

# Hedef Değişkenin Kategorik Değişkenler ile Analizi #

df.groupby("sex")["survived"].mean()
df.groupby("alone")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target_Mean": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df, "survived", "sex")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)


# Hedef Değişkenin Numerik Değişkenler ile Analizi #

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age": ["mean", "sum"]})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "survived", "age")


#########################################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#########################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv(r"C:\Users\ahmet\OneDrive\Masaüstü\breast-cancer.csv")
df = df.iloc[:, 2:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize':(12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# Yüksek Korelasyonlu Değişkenlerin Silinmesi #

cor_matrix = df.corr().abs()

upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

drop_list = [col for col in upper.columns if any(upper[col] > 0.90)]

cor_matrix[drop_list]
df.drop(drop_list, axis=1)

def high_corr(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper.columns if any(upper[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_corr(df)
high_corr(df.drop(drop_list, axis=1), plot=True)


