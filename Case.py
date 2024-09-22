import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import datetime as dt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 400)

df2 = pd.read_excel(
    "Datasets/side_effect_data_1.xlsx",
    index_col="Kullanici_id")
df = df2.copy()
df.head()
df.shape
df.info


###############################
def grab_col_names(dataframe, cat_th=10, car_th=30):
    """
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken
            sayısı

        """

    # cat_cols, cat_but_car
    cat_cols = [
        col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique(
    ) < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique(
    ) > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [
        col for col in dataframe.columns if dataframe[col].dtypes in [
            "int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # date_cols
    date_cols = [
        col for col in dataframe.columns if dataframe[col].dtypes == "datetime64[ns]"]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f"date_cols: {len(date_cols)}")
    return cat_cols, num_cols, cat_but_car, date_cols


cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "Kullanici_id"]
cat_cols = [col for col in cat_cols if col not in "Uyruk"]

#####################################
msno.bar(df, color="#75658e", figsize=(12, 6), fontsize=10)


def missing_values_table(dataframe, na_name=False):
    na_columns = [
        col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (
        dataframe[na_columns].isnull().sum() /
        dataframe.shape[0] *
        100).sort_values(
        ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)],
                           axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(
    ), "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=90)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=15)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

########################## Yeni değişkenler ver Çaprazlamalar ############

suan = dt.datetime.now()
df["Yas"] = df["Dogum_Tarihi"].apply(
    lambda x: suan.year - x.year - ((suan.month, suan.day) < (x.month, x.day)))
df["Kategorik_Yas"] = pd.cut(df["Yas"],
                             bins=[0, 18, 35, 65, np.max(df["Yas"])],
                             labels=["Çocuk", "Genç", "Orta Yaşlı", "Yaşlı"])

df["Kategorik_Yas"].value_counts()

###VKI###
# eksik değerleri doldurduktan sonra yapmak gerekiyor
df["VKI"] = df["Kilo"] / (df["Boy"] / 100) ** 2
df["Kategorik_VKI"] = pd.cut(
    x=df["VKI"],
    bins=[
        0,
        18.5,
        24.9,
        29.9,
        39.9,
        (df["VKI"].max())],
    labels=[
        "Zayıf",
        "Normal Kilolu",
        "Fazla Kilolu",
        "Obez",
        "Aşırı Obez"])

df.groupby("Kategorik_VKI").agg({"Yan_Etki": "count"})

################ Yas Gruplarına Göre İlaç Yan Etkileri ###################

yas_yan_etki = df.groupby(["Kategorik_Yas", "Yan_Etki"]
                          ).size().unstack(fill_value=0)
top5_yan_etki = df["Yan_Etki"].value_counts(ascending=False).head(5).index
filtreed_yan_etki = yas_yan_etki[top5_yan_etki]

filtreed_yan_etki.plot(kind="bar")
plt.xlabel("Yan Etkiler")
plt.ylabel("Frekans")
plt.title("Yaş Gruplarına Göre Yaygın İlaç Yan Etkileri")
plt.xticks(rotation=45)
plt.legend(title="Yaş Grubu")
plt.tight_layout()

# Cinsiyete Göre Yan Etki ##########

cinsiyet_yan_etki = df.groupby(
    ["Cinsiyet", "Yan_Etki"]).size().unstack(fill_value=0)
filtreed_yan_etki = cinsiyet_yan_etki[top5_yan_etki]

filtreed_yan_etki.plot(kind="bar")
plt.xlabel("Yan Etkiler")
plt.ylabel("Frekans")
plt.title("Cinsiyete Göre Yaygın İlaç Yan Etkileri")
plt.xticks(rotation=45)
plt.legend(title="Cinsiyet")
plt.tight_layout()

######## VKI Göre Yan Etki ########
vki_yan_etki = df.groupby(["Kategorik_VKI", "Yan_Etki"]
                          ).size().unstack(fill_value=0)
filtreed_yan_etki = vki_yan_etki[top5_yan_etki]

filtreed_yan_etki.plot(kind="bar")
plt.xlabel("Yan Etkiler")
plt.ylabel("Frekans")
plt.title("VKI Göre Yaygın İlaç Yan Etkileri")
plt.xticks(rotation=45)
plt.legend(title="VKI")
plt.tight_layout()
#### Kan grubuna Göre Yan Etki ####

kangrubu_yan_etki = df.groupby(
    ["Kan Grubu", "Yan_Etki"]).size().unstack(fill_value=0)
filtreed_yan_etki = kangrubu_yan_etki[top5_yan_etki]

filtreed_yan_etki.plot(kind="bar")
plt.xlabel("Yan Etkiler")
plt.ylabel("Frekans")
plt.title("VKI Göre Yaygın İlaç Yan Etkileri")
plt.xticks(rotation=45)
plt.legend(title="VKI")
plt.tight_layout()
##### En yaygın 5 Alerjide Görülen en sık 5 Yan Etki #####
en_yaygin_alerjiler = df["Alerjilerim"].value_counts(
    ascending=False).head(5).index
filtered_df = df[df["Alerjilerim"].isin(en_yaygin_alerjiler)]
alerji_yan_etki = filtered_df.groupby(
    ['Alerjilerim', "Yan_Etki"]).size().unstack(fill_value=0)
filtreed_yan_etki = alerji_yan_etki.loc[:, top5_yan_etki]

filtreed_yan_etki.plot(kind="bar", stacked=True)
plt.xlabel("Alerjiler")
plt.ylabel("Frekans")
plt.title("En Yaygın 5 Alerjiye Göre En Sık Görülen 5 Yan Etki")
plt.xticks(rotation=45)
plt.legend(title="Yan Etkiler")
plt.tight_layout()
plt.show()
### Kronik Hastalıkları Tekilleştirme ve Yan Etkilere Göre Gruplama ###
df_kronik = df.copy()
df_kronik["Kronik Hastaliklarim"] = df_kronik["Kronik Hastaliklarim"].str.split(", ")
df_kronik = df_kronik.explode("Kronik Hastaliklarim")
kronik_hastalik_tekil_sorted = df_kronik["Kronik Hastaliklarim"].value_counts(ascending=True)

kronik_hastalik_tekil_sorted.plot(
    kind="barh", color=[
        "lightblue", "lightcoral"])
plt.xlabel("Frekans")
plt.ylabel("Kronik Hastalıklar")
plt.title("Kronik Hastalıkların Dağılımı")
plt.tight_layout()
plt.show()
#### Gruplama ve Grafik####

kronik_yan_etki = df_kronik.groupby(
    ["Kronik Hastaliklarim", "Yan_Etki"]).size().unstack(fill_value=0)
filtreed_yan_etki = kronik_yan_etki.loc[:, top5_yan_etki]
filtreed_yan_etki.plot(kind="bar", stacked=True)
plt.xlabel("Kronik Hastalıklar")
plt.ylabel("Frekans")
plt.title("Kronik Hastalıklara Göre Görülen En yaygın 5 Yan Etki")
plt.xticks(rotation=45)
plt.legend(title="Yan Etkiler")
plt.tight_layout()
plt.show()

###############################
df["Ilac_Adi"].value_counts()
kremler = df[df["Ilac_Adi"].str.contains("cream")]
kremler["Yan_Etki"].value_counts()  # Problem mevcut

###################################################################
# Eksik Veri Problemlerini Çözme

msno.bar(df, color="#75658e", figsize=(12, 6), fontsize=10)

### Tekilleştirme Problemi Çözümü ###
kronik_hastaliklar = [
    "Kronik Hastaliklarim",
    "Baba Kronik Hastaliklari",
    "Anne Kronik Hastaliklari",
    "Kiz Kardes Kronik Hastaliklari",
    "Erkek Kardes Kronik Hastaliklari"]
df[kronik_hastaliklar] = df[kronik_hastaliklar].apply(
    lambda col: col.str.split(", "))

for col in kronik_hastaliklar:
    df = df.explode(col)

# Tekilleştirme Problemi Çözüldükten Sonra Kategorik Verilerin Doldurulması
eksik_cat = [
    "Alerjilerim",
    "Kronik Hastaliklarim",
    "Baba Kronik Hastaliklari",
    "Anne Kronik Hastaliklari",
    "Kiz Kardes Kronik Hastaliklari",
    "Erkek Kardes Kronik Hastaliklari"]
eksik_bil = ["Cinsiyet", "Kan Grubu"]

# Hastalıklar için
for col in eksik_cat:
    df[col] = df[col].fillna("Bilinen Hastalık Yok")
# Bilinmeyen değerler için
for col in eksik_bil:
    df[col] = df[col].fillna("Bilinmiyor")
# İl Değişkeni için en çok tekrar eden değerle doldurma işlemi
df["Il"] = df["Il"].fillna(df["Il"].mode()[0])

# Sayısal Değerler için KNN Algoritması ile tahmine dayalı doldurma işlemi
na_num_cols = ["Kilo", "Boy"]
imputer = KNNImputer(n_neighbors=5)
df[na_num_cols] = imputer.fit_transform(df[na_num_cols])

# İlgili İşlemler yapıldıktan sonra Yeniden VKI hesaplama işleminin yapılması
df["VKI"] = df["Kilo"] / (df["Boy"] / 100) ** 2
df["Kategorik_VKI"] = pd.cut(
    x=df["VKI"],
    bins=[
        0,
        18.5,
        24.9,
        29.9,
        39.9,
        (df["VKI"].max())],
    labels=[
        "Zayıf",
        "Normal Kilolu",
        "Fazla Kilolu",
        "Obez",
        "Aşırı Obez"])

# Eksik Verimiz Var mı Kontrol Edelim

df.isnull().sum().any()
msno.bar(df, color="#75658e", figsize=(12, 6), fontsize=10)

# Son durumda kategorik ve numerik veriler
cat_cols, num_cols, cat_but_car, date_cols = grab_col_names(df)


# One Hot Encoder İşlemi

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe,
        columns=categorical_cols,
        drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape

# Numerik değişkenler için standartlaştırma işlemi

ss = StandardScaler()

df[num_cols] = ss.fit_transform(df[num_cols])
