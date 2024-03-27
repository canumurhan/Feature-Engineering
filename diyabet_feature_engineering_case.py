#İş Problemi

##################################################################
#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek
#bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli
#olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

###################################################################

#Veri Seti Hikayesi
#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri
#setinin parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan
#21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan
#verilerdir.
#Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise
#negatif oluşunu belirtmektedir.

###################################################################

#Pregnancies:Hamilelik sayısı
#Glucose:Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
#Blood Pressure:Kan Basıncı (Küçük tansiyon) (mm Hg)
#SkinThickness:Cilt Kalınlığı
#Insulin:2 saatlik serum insülini (mu U/ml)
#DiabetesPedigreeFunction:Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
#BMI:Vücut kitle endeksi
#Age:Yaş (yıl)
#Outcome:Hastalığa sahip (1) ya da değil (0)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler,RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width',500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: "%.1f" %x)

df=pd.read_csv('Week 6 Feature Engineering/diabetes.csv')


#Görev 1 : Keşifçi Veri Analizi

#Adım 1: Genel resmi inceleyiniz.

df.head()
df.describe().T
df.isnull().sum()
df.value_counts().sum()
df.isnull().sum().sum()

#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "object"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != "object"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes != "object"]
    return cat_cols, num_but_cat, cat_but_car

cat_cols, num_but_cat, cat_but_car = grab_col_names(df)

num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in num_but_cat]

print(f"Observations: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print(f'cat_cols: {len(cat_cols)}')
print(f'num_cols: {len(num_cols)}')
print(f'cat_but_car: {len(cat_but_car)}')
print(f'num_but_cat: {len(num_but_cat)}')


#Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

df.corr()

#Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin
#ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

num_target_mean=df.groupby("Outcome").mean()

#Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

outlier_results = [(col, check_outlier(df, col)) for col in df.columns]

#Adım 6: Eksik gözlem analizi yapınız.

df.isnull().values.any()
df.isnull().sum()


#Adım 7: Korelasyon analizi yapınız.

df.corr()

#Görev 2 : Feature Engineering

#Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde
#eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren
#gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin
#glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır
#değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere
#işlemleri uygulayabilirsiniz.

def handle_missing_and_outliers(df, outlier_results):
    # Eksik değerleri NaN ile değiştir
    df.replace(0, np.nan, inplace=True)

    # Aykırı değerleri düzelt
    for col, is_outlier in outlier_results:
        if is_outlier:
            low_limit, up_limit = outlier_thresholds(df, col)
            df.loc[(df[col] < low_limit), col] = low_limit
            df.loc[(df[col] > up_limit), col] = up_limit


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

df = pd.get_dummies(df, columns=num_cols, drop_first=True)

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)





#Adım 2: Yeni değişkenler oluşturunuz.
# Vücut Kitle Endeksi ve Yaşın Çarpımı
df['BMI_Age_Product'] = df['BMI'] * df['Age']

# Kan Basıncı ve Yaşın Çarpımı
df['BloodPressure_Age_Product'] = df['BloodPressure'] * df['Age']

# Gebelik ve Yaş Farkı
df['Pregnancies_Age_Difference'] = df['Pregnancies'] - df['Age']

# İnsülin ve Glikozun Çarpımı
df['Insulin_Glucose_Product'] = df['Insulin'] * df['Glucose']

# Diyabet Pedigri Fonksiyonu ve Yaşın Toplamı
df['DPF_Age_Sum'] = df['DiabetesPedigreeFunction'] + df['Age']

# BMI ve Cilt Kalınlığı Oranı
df['BMI_SkinThickness_Ratio'] = df['BMI'] / df['SkinThickness']

df.head()

#Adım 3: Encoding işlemlerini gerçekleştiriniz.
 binary_cols=[col for col in df.columns if df[col].dtypes not in [int,float] and df[col].nunique()==2]
 for col in binary_cols:
     label_enc