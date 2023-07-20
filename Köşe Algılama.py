import cv2    # Gerekli Kütüphaneler İçeri aktarılır.
import matplotlib.pyplot as plt
import numpy as np

# siyah beyaz resmi içe aktarma.
img = cv2.imread("sudoku.jpg", 0)

# Resimi ondalıklı sayılara çeviriyoruz.
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")


# cv2.cornerHarris() fonksiyonunu kullanarak görüntünün kenarlık haritası hesaplanır.
# blockSize, kare bloklarının boyutudur. ksize,türetilmiş türev filtresinin boyutudur, bu filtre kenarlık haritasının hesaplanmasında kullanılır.
# k, kenar hassasiyeti parametresidir. bu değer algılama sonucunu ayarlar
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")


# cv2.dilate() fonksiyonu kenarlık haritası genişletilir. Bu işlem kenarların daha belirgin hale gelmesini sağlar.
# ikinici parametrede varsayılan bir genişletme kullanacağından bahseder, buraya none diyoruz.
dst = cv2.dilate(dst, None)

# belirli bir eşik değeri üzerindeki pikseller giriş görüntüsünde beyaz(1) olarak işaretlenir
# bu sayede köşe noktalar vugulanmış olur.
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")


# shi tomasi detection algoritması
img = cv2.imread("sudoku.jpg", 0)

# algoritma ondalıklı sayı tipini kullanarak hesaplamalar yapar.
img = np.float32(img)

# cv2.goodFeaturesToTrack() fonksiyonu kullanılarak köşeler tespit edilir.
# girdileri = görsel, tespit edilecek köşe sayısı, köşe kalite faktörü(daha yüksek bir değer daha yüksek kaliteli köşelerin tespit edilmesini sağlar)
# son parametre girdisi ise köşe noktalarının minimum Euclidean mesafesi.
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)
corners = np.int64(corners)   # tespit edilen noktalar bir numpy dizisi şeklinde aktarılır.


for i in corners:
    x,y = i.ravel()   # ravel() fonskyionu köşe noktaların kordinatlarını alır ve x,y değişkenlerine aktarır.
    cv2.circle(img, (x,y),3,(125,125,125),cv2.FILLED)   # kordinatları belli olan her köşe nokatsında bir daire çizilir.
    
plt.imshow(img)
plt.axis("off")

