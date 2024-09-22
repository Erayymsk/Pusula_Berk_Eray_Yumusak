# Data Science Case Study: Drug Side Effects Analysis 

Berk Eray Yumuşak \
eraayp@gmail.com


## Genel Bakış
### Proje Özeti 
Bu proje, ilaçların yan etkilerine ilişkin bir veri seti üzerinde kapsamlı bir veri analizi yapmayı ve bu veriyi tahminsel modeller için hazırlamayı hedeflemektedir. Yan etkiye neden olan faktörleri keşfetmek ve bu faktörleri daha iyi anlayarak modelleme sürecine dahil etmek projenin başlıca amaçlarındandır. 

Projenin ana adımları şunlardır:

#### Exploratory Data Analysis (EDA):
Veri setinin genel yapısını anlamak, dağılımları ve anomalileri tespit etmek amacıyla Pandas, Matplotlib ve Seaborn kütüphaneleri kullanılarak görselleştirme teknikleri uygulanmıştır. Bu aşamada, veri içindeki değişkenler arasındaki ilişkiler ve olası yan etkiler arasındaki bağlantılar incelenmiştir.
#### Veri Ön İşleme:
EDA aşamasında tespit edilen eksik veya hatalı verilerin düzeltilmesi, kategorik değişkenlerin kodlanması ve numerik verilerin ölçeklendirilmesi gibi veri temizleme işlemleri gerçekleştirilmiştir. Bu sayede, modelleme aşaması için temiz ve güvenilir bir veri seti oluşturulmuştur.
#### Modelleme için Hazırlık:
Veriyi tahminsel modellemeye uygun hale getirecek adımlar uygulanmıştır. Bu aşamada veri, uygun formatlarda ve standartlarda düzenlenmiş ve ileride kullanılacak makine öğrenimi modelleri için hazır hale getirilmiştir.


## Yükleme 

```bash 
  git clone https://github.com/Erayymsk/Pusula_Berk_Eray_Yumusak.git # Clone
 ```

```bash 
  cd Pusula_Berk_Eray_Yumusak
```
```bash 
  pip install -r requirements.txt  # install
```
## Kullanılan Kütüphaneler ve Metotlar
* Python
* Pandas
* Numpy
* Seaborn
* Matplotlib
* KNNImputer
* StandartScaler
* Missingno 
## Sonuçlar
Bu proje kapsamında, ilaç yan etkilerine neden olabilecek faktörler detaylı bir şekilde analiz edilmiştir. Yapılan analizler, yan etkilerin sadece ilaçların kendisiyle sınırlı kalmadığını, hasta yaşı, cinsiyeti, sağlık durumu gibi çeşitli etkenlerin de bu etkilerin oluşmasında önemli rol oynadığını göstermiştir.

Veri ön işleme ve keşifsel veri analizi sonucunda, veriyi daha ileri tahminsel modeller için uygun hale getirmek adına önemli adımlar atılmıştır. Detaylı bulgular ve analiz sonuçları ilgili raporda mevcuttur.
