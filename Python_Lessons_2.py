# Veri Görselleştirme: Matplotlib & Seaborn #

# Matplotlib

# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: hist, boxplot

# Kategorik değişken görselleştirme #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind = "bar")
plt.show()

# Sayısal değişken gösterme #
plt.hist(df["age"])
plt.show()  # age değerlerinin histogram grafiği

plt.boxplot(df["fare"])
plt.show()  # fare değerlerinin box grafiği


# Matplotlib'in Özellikleri #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#plot

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y, 'o-')
plt.show()


# marker
# matplotlib.markers website - https://matplotlib.org/stable/api/markers_api.html
y = np.array([13,28,11,100])
plt.plot(y, marker = '*')
plt.show()

# hatalarla uğraş
# araştır
# bilgi

#Line#
y = np.array([13,28,11,100])
plt.plot(y, linestyle = "dashed")
plt.show()

plt.plot(y, linestyle = "dotted")
plt.show()

plt.plot(y, linestyle = "dashdot", color = "red")
plt.show()


# multiple lines #
x = np.array([23,18,31,10])
y = np.array([13,28,11,100])
plt.plot(x)
plt.plot(y)
plt.show() # bu ifade önemli alışkanlık edin



# Labels #

x = np.array([23,18,31,10])
y = np.array([13,28,11,100])
plt.plot(x, y)
#başlık
plt.title("bu ana başlık")

# x eksenini isimlendirme
plt.xlabel("x ekseni")
# y eksenini isimlendirme
plt.ylabel("y ekseni")
plt.grid()
plt.show()

# Subplots #

#plot 1
x = np.array([80,85,90,95,100,105,110,115,120])
y = np.array([240,250,260,270,280,290,300,310,320])
plt.subplot(1,2,1)
plt.title("plot 1")
plt.plot(x, y)


#plot 2
x = np.array([75,85,90,95,100,105,110,115,120])
y = np.array([24,250,260,270,280,290,300,310,320])
plt.subplot(1,2,2)
plt.title("plot 2")
plt.plot(x, y)
plt.show()

# matplotlib her şeyin atasıdır.

# Seaborn #

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()
df["sex"].value_counts()
sns.countplot(x="sex", data=df, color="red")
plt.show()

df['sex'].value_counts().plot(kind='bar')
plt.show()


# Sayısal değişken Görselleştirme

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()