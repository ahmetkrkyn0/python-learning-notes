from numpy.ma.core import append

# Sanal ortamların listelenmesi
# conda env list

# sanal ortam oluşturma
# conda create -n myenv


# To activate this environment, use
#
#     $ conda activate myenv
#
# To deactivate an active environment, use
#
#     $ conda deactivate



# conda list ( paketleri listeler )



# conda install numpy ( paket yükleme )



# aynı anda birden fazla paket yükleme
# conda install numpy scipy pandas





# paket silme:
# conda remove package_name




# Belirli bir versiyona göre paket yükleme:
# conda install numpy=1.20.1




# paket yükseltme
# conda upgrade conda



# tüm paketlerin yükseltilmesi
# conda upgrade -all


# pip: pypi ( python package index) paket yönetim  aracı
# paket yükleme
# pip install pandas


# versiyona göre paket yükleme
# pip install pandas==1.2.1


#Sayılar: integer

x = 46
type(x)

#Sayılar: float
y = 10.2
type(y)

#Sayılar: complex
x = 2j + 1
type(x)

#String

a = "Merhabalar"
type(a)

#boolean
True
False
type(True)
5 == 4
type(3==2)


#liste
x = ["btc", "eth", "xrp"]
type(x)


#sözlük
x = {"name": "Peter", "Age": 36}
type(x)


#tuple
x = ("python", "ml", "ds")
type(x)


#set
x = {"name","Peter","Age"}
type(x)



a = 2
a ** 2
a ** 3

float(a)

"John"

name  = "John"
print(name)



#eğer uzun string yapmak istersek alt satırlara indirmek için """ ..... """ kullanılır long_str



name[0]  # ilk harfi yazdırma



# karakter dizilerinde slice işlemi
name[0:2] # sıfırdan başla 2 ye kadar git !!



#string içinde eleman sorgulamak
"veri" in name
"John" in name
print("baba\nbaba")


#String metodları

dir(int)

####
#len
####
name = "John"
type(name)
type(len)
len(name) # karakter sayısını yazdırır

#eğer bir fonk class yapısı içinde tanımlandıysa method denir, tanımlanmadıysa fonk denir.

"miuul".upper() # karakter dizilerini büyütmek için kullanlılır
"MİEUUL".lower() # karakter dizilerini küçültmek için kullanılır


hi = "Hello AI Era"
hi.replace("l", "p")  #değişitrmek istediğimiz harfi yazıp yeni harfi yazıyoruz harfler değişiyor


"Hello AI Era".split() #cümleyi böler

" ofofo ".strip()
"ofofo".strip("o") #soldan ve sağdan kırpar

#ilk harfelri büyütmek için
"foo".capitalize()

dir("foo")

"foo".startswith("o")




# Liste(List)

# - değiştirilebilir, sıralıdır, index işlemleri yapılabilir.
# - kapsayıcıdır.

notes = [1,2,3,4,5,6]
type(notes)

notes[0]
notes[0:5]
notes[0:6]

# Liste Metodları

dir(notes)

len(notes)
notes.append(100)
notes

notes.pop(0)
notes

notes.insert(0,100)
notes
len(notes)

# Sözlük ( Dictionary )
# Değiştirilebilir.
# Sırasız. (3.7'den sonra sıralı.)
# Kapsayıcı

# key-value

dictionary = {"REG": "REgression", "BAB": "Babanne"}
dictionary["REG"]

"REG" in dictionary

dictionary.get("REG")

dictionary["REG"] = ["YSA", 10]
dictionary.get("REG")

dictionary.keys()
dictionary.values()
dictionary.items()

dictionary.update({"REG": 11})


#Demet (Tuple)
# - Değiştirilemez
# - Sıralıdır
# - Kapsayıcıdır

t = ("john", "mark", 1,2)
t[0]


# Set
# - Değiştirilebilir
# - Sırasız + Eşsizdir
# - Kapsayıcıdır

#difference(): iki kümenin farkıdır

set1 = set([1,2,4,5,6])
set2 = set([1,2,4,5,7])

set1.difference(set2) # 6
set2.difference(set1) # 7

set1.symmetric_difference(set2) # 6,7

set1.intersection(set2) # 1,2,4,5

set1 & set2 # 1,2,4,5
set1 - set2 # 6

set1.union(set2) # birleştirir

# isdisjoint(): iki kümenin kesişimi boş mu?
set1.isdisjoint(set2) #false

# issubset(): bir küme diğer kümenin alt kümesi mi ?
set1.issubset(set2) # False

#  issuperset(): Bir küme diğerini kapsıyor mu?



#########################
#Fonksiyonlar
#########################

def summer(arg1, arg2):
    """
    Parameters
    ----------
    arg1
    arg2

    Returns
    -------

    """
    print(arg1 + arg2)

summer(1, 2)

def say_hi(string):
    print(string)
    print("hello")
    print("hi again")

    say_hi("baba")

def multiplier(a, b):
    c = a * b
    print(c)

multiplier(9, 2)


# girilen değerleri listede saklayan fonk

list_store = []

def add_element(x, y):
    c = x * y
    list_store.append(c)
    print(list_store)

add_element(1, 8)
add_element(2, 8)
add_element(3, 8)



def divide(a, b=1):
    print(a / b)

divide(3)


############
# Ne Zaman Fonksiyon Yazma İhtiyacımız Olur?
############

# varm, moisture, charge

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

calculate(15, 5, 1)


######################
# Return: Fonksiyonun Çıktılarını Girdi Olarak Kullanmak
######################


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge

    return  varm, moisture, charge, output

type(calculate(15, 5, 1)) ## Tuple

varm, moisture, charge, output = calculate(15, 5, 1)

kilo, boy, indeks, sonuc = calculate(15, 5, 20)



############################################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
############################################

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge) # int yazarak verilen çıktının int değer olmasını sağlarız

calculate(15, 5, 3)

def standardization(a, p):
    return int(a * 10 / 100 * p * p)

def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b)

all_calculation(15, 5, 3, 90, 2)



##############################################
# Local & Global Değişkenler
##############################################

list_store = [1, 2]
type(list_store)


def add_element(g, h):
    j = g * h
    list_store.append(j)
    print(list_store)

add_element(1, 2)
add_element(3, 4)


###############################
# If ( Conditions )
###############################

# True - False
1==1 # true
1==2 # false

#if
if 1 == 2:
    print("True")
else: print("False")

num = 3
if num == 3:
    print("True")
else:
    print("False")

num = int(num)
def number_check(num):
    if num == 3:
        print("True")
    elif num == 5:
        print("True")
    else:
        print("False")

number_check(4.6)
number_check(1)
number_check(4)
number_check(5)


#####################################################
# DÖNGÜLER ( LOOPS )
####################################################
# for loop

students = ["John", "Mark", "Venessa", "Mariam"]
students[0] # John

for student in students:
    print(student.upper())  #her harfi büyütüp her kelimeyi tek tek yazdırdık

salaries = [34, 56, 78]
for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary * 20 / 100 + salary))

def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)

new_salary(34, 56)
new_salary(100000, 78)

for salary in salaries:
    print(new_salary(salary, 10))

for salary in salaries:
    if salary <= 50:
        print(new_salary(salary, 50))
    else: print(new_salary(salary, 10))


################################################
# Uygulama - Mülakat Sorusu
################################################

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.
# before: "hi my name is john and i am learning python"
# after def: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

#ex
range(len("miuul"))
range(0, 5)
for o in range(len("miuul")):
    print(o)


def alternating(string):
    s  = string
    new_string = ""
    # girilen string'in indexlerinde gez
    for string_index in range(len(string)):
        # index çiftse büyük harfe çevir
        if string_index % 2 == 0:
            new_string += s[string_index].upper()
        # index tekse küçük harfe çevir
        else:
            new_string += s[string_index].lower()
    print(new_string)


alternating("miuul") #MiUuL




####################################
# break & continue & while
####################################


#break

salaries = [34, 56, 78]

for salary in salaries:
    if salary == 78:
        break
    print(salary)


# continue
for salary in salaries:
    if salary == 56:
        continue
    print(salary)


#while
num1 = 1
while num1 < 10:
    print(num1)
    num1 += 1



###################################################
# Enumerate: Otomatik Counter/Indexer ile for loop
###################################################


names = ["John", "Mark", "Venessa", "Mariam"]

A = []
B = []
C = []

for index, name in enumerate(names):
    if index % 2 == 0:
        C.append(name)
    else:
        B.append(name)


#########################################
# Uygulama - Mülakat Sorusu
#########################################
# divide_student fonksiyonunu yazınız
# Çift indexte yer alan öğrencileri bir listeye alınız.
# Tek indexte yer alan öğrencileri başka bir llisteye alınız.
# Fakat bu liste tek bir liste olarak return olsun.

games= ["rdr", "gta", "pes", "fifa", "cs"]

def divide_games(games_list):
    new_games_even = []
    new_games_odd = []
    for i, game in enumerate(games):
        if i % 2 == 0:
           new_games_even.append(game)
        else:
           new_games_odd.append(game)
    return [new_games_even, new_games_odd]

divide_games(games)


def divide_students(students_list):
    groups = [[], []]
    for index, student in enumerate(students_list):
        if index % 2== 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    return groups

students_1 = ["anne", "baba", "climb"]
st = divide_students(students_1)
st[1]



###################################################
# alternating fonksiyonunun enumerate ile yazılması
###################################################

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")



################################################
# Zip
################################################

students = ["John", "Mark", "Venessa", "Mariam"]
departments = ["math", "science", "statistics", "astronomy"]
ages = [18, 19, 20, 20]

list(zip(students, departments, ages)) # ex -> ('John', 'math', 18)


##################################
# lambda, map, filter, reduce
##################################

def summer(a, b):
    return a + b

summer(1, 2) * 9

new_sum = lambda q, b: q + b  # like def, kullan at fonksiyondur
new_sum(1, 2) * 9


#map
salaries = [34, 56, 78]

def new_salary(x):
    return int(x * 10 / 100 + x)
new_salary(10000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries)) #ex -> [37,61,85] like for
list(map(lambda x: x * 10, salaries))

#filter

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store)) # like if

#reduce
from functools import reduce
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
reduce(lambda x, y: x + y, list_store)  # output: 55

## break ifadesi sadece döngü içinde kullanılır ex for, while



######################################
# COMPREHENSIONS
######################################

#################################
# List Comprehension
#################################

salaries = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

def new_salary(x):
    return int(x * 10 / 100 + x)

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))   # artırılmış maaşlar başka bir listeye ekleniyor

print(null_list)


[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries] # kısaca

[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000] # if tek başına kullanırılken en sağda olur

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries] # if, else ile birliikte kullanılırken solda olur

worker_names = ["John", "Mark", "Venessa", "Mariam"]

worker_names_1 = ["John", "Venessa"] #istenmeyenleri küçült kalanları büyült

[worker.lower() if worker in worker_names_1 else worker.upper() for worker in worker_names]

[worker.upper() if worker not in worker_names_1 else worker.lower() for worker in worker_names]

# tek bir satırda tüm ilemleri yaptık ve satır fazlalığından kurtulduk


############################
# Dict Comprehension
############################

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4,
              'e': 5,
              'f': 6 }

dictionary.keys()
dictionary.values()
dictionary.items()
list(dictionary.values())
list(dictionary.items())
list(dictionary.keys())

{k:v ** 2 for (k, v) in dictionary.items()} #value değerleri üzerinde işlem yaptık

{k.upper():v for (k, v) in dictionary.items()} #key değerleri üzerinde işlem yaptık


#########################
# Uygulama - Mülakat Sorusu
###########################
# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir
# Key'ler orjinal değerler value'lar ise değiştirilmiş değerler olacak


numbers = range(10)
new_dict = {}

for number in numbers:
    if number % 2 == 0:
        new_dict[number] = number ** 2
    else:
        new_dict[number] = number

print(new_dict)

{number: number ** 2 for number in numbers if number % 2 == 0}


######################################
# List&Dict Comprehension Uygulamalar #
######################################

#####################################################
# Bir veri setindeki değişken isimlerini değiştirmek
#####################################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

A = []
for col in df.columns:
    A.append(col.upper())

print(A)

df.columns = A

df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]

#####################################################################################
# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
#####################################################################################

df = sns.load_dataset("car_crashes")
df.columns = ["FLAG_" + col.upper() if col[0:3] == "ins" else col.upper() + "_FLAG" for col in df.columns]
df.columns

df = sns.load_dataset("car_crashes")
df.columns = ["FLAG_" + col.upper() if "ins" in col else "NO_FLAG_" + col.upper() for col in df.columns]
df.columns


##################################################################################
# Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
##################################################################################
##########################################################
###{
###'total': ['mean', 'min', 'max', 'var'],
###'speeding': ['mean', 'min', 'max', 'var'],
###'alcohol': ['mean', 'min', 'max', 'var'],
###'not_distracted': ['mean', 'min', 'max', 'var'],
###'no_previous': ['mean', 'min', 'max', 'var'],
###'ins_premium': ['mean', 'min', 'max', 'var'],
###'ins_losses': ['mean', 'min', 'max', 'var']
###}
#########################################################
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

#1.step
my = {}
ex_list = ["mean", "min", "max", "var"]
my = {col: ex_list for col in df.columns}
print(my)

#2.step
num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] = agg_list


df[num_cols].head()
df[num_cols].agg(soz)
