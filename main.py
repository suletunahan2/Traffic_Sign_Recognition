
import pickle 
'''pickle modülü, Python nesnelerini serileştirmenizi sağlayan ufak ama güçlü bir kütüphane. Kullanımı oldukça basit. Tek dikkat etmeniz gereken şey, yazdıracağınız dosyayı ikili (binary) formatta kullanmanız.
Bir python nesnesinin byte stringine dönüşmüş  haline pickle denir. ''' 
import numpy as np
import csv #excelden okumak için
import matplotlib.pyplot as plt # veri görsellestirme icin
import random


# Step 1: Load The Data

training_file = 'C:\\Users\\suletunahan2\\Desktop\\tüm_dersler_2020\\softwarelab2\\dataset\\traffic-signs-data\\train.p'
testing_file = 'C:\\Users\\suletunahan2\\Desktop\\tüm_dersler_2020\\softwarelab2\\dataset\\traffic-signs-data\\test.p'

with open(training_file,mode='rb') as file:  # rb:Opens a file for reading only in binary format. The file pointer is placed at the beginning of the file. This is the default mode.
    train = pickle.load(file)
with open(testing_file, mode='rb') as file:
    test = pickle.load(file)

# print(type(train), type(test)) # Burda verilerin dictionary oldugunu görüyoruz.

signs=[] 
# Comma Separated Values (CSV) – Virgülle Ayrılmış Değerler  dosyası, bir veri listesi içeren düz metin dosyasıdır. 
with open('signnames.csv', 'r') as csvfile: # her bir label id sini adla eşleştiren excel dosyasını okuyoruz ve listeye atıyoruz.
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None) # signnamesi listeye atma
    for row in signnames:
        signs.append(row[1])
    csvfile.close()

# print(signs)


#Step 2: Dataset Summary & Exploration

# print(train)

''' Eger dataya bakacak olursak :

data bir dictionary dir ve 4 key-value çiftinden olusur. Bunlar:
'features' , 'labels' , 'sizes ' , 'coords' dir.

'features' : görüntülerin pixel verilerini içeren 4 boyutlu dizidir.(num examples,width,height,channels)
'labels' : trafik isaretlerinin class\label yani etiket id sini iceren 1 boyutlu dizidir. signnames.csv dosyası her id icin ad eşlemesi yapıyor.
'sizes ' : görüntünün orjinal genişlik ve yüksekilgini içeren tuple
'coords' : koordinatlarını temsil eden tuples (x1, y1, x2, y2) içeren bir listedir.

 '''

# İlk basta x ve y ye feature ve label datalarını atıyoruz.
x_train=train['features']
y_train = train['labels']

x_test=test['features']
y_test =test['labels']

# print(y_test) #output is label like [16  1 38 ...  6  7 10]
# .shape →Numpy array nesnesinin kaç satır ve sütundan oluştuğunu gösteren bir tupple nesnesi döndürür.
print(f"Number of examples for training: {x_train.shape[0]}")
print(f"Number of examples for test : {x_test.shape[0]}")
print(f"Image data shape : {x_train[0].shape}")
print(f"Number of  classes : {len(np.unique(y_train))}")


# matplotlib

def image_list(dataset_x,dataset_y, ylabel="",cmap=None):
    '''Bir döngü sayesinde datasetten rastgele üretilen bir sayiya gore verileri görselleştirir.
    Bunu yapmak icin matplotlib kütüphanesini kullanır.
    Parametreler :(dataset_x : x_train, dataset_y : y_train, ylabel : y eksenin ismi ,cmap : color map (renk özellestirme icin))
    Matplotlib işlevi imshow (), 2 boyutlu bir numpy dizisinden bir görüntü oluşturur. Resimde dizinin her bir öğesi için bir kare olacaktır. Her karenin rengi, karşılık gelen dizi öğesinin değeri ve imshow () tarafından kullanılan renk haritasına göre belirlenir.

    '''
    plt.figure(figsize=(15,16)) # 15 inç genişlikte ve 16 inç yükseklikte bir şekil yaratır.

    for i in range(5):
        plt.subplot(1, 5, i+1) # subplot
        indx = random.randint(0, len(dataset_x)) # 0 ile datasaet boyutu arası rastgele int sayi üret.
        cmap = 'gray' if len(dataset_x[indx].shape) == 2 else cmap #  cmap: color map , used to specify colors
        plt.imshow(dataset_x[indx], cmap = cmap) # (numpy array generating the image , cmap)

        plt.xlabel(signs[dataset_y[indx]]) # xlabel : sign listemizden gelir.
        plt.ylabel(ylabel)

        # Dikey ve yatay kılavuz çizgilerinin yerlerini elle belirlemek için sırayla xticks ve yticks kullanılır. 
        # Bunlar argüman olarak birer sayı dizisi alırlar. 

        plt.xticks([])
        plt.yticks([])

    # plt.tight_layout : alt grafik parametrelerini şekil alanına sığacak şekilde otomatik olarak alt grafik parametrelerini ayarlar.
    plt.tight_layout()
    plt.show() # displays the plot


# Plotting sample examples

image_list(x_train, y_train, "Training example")
image_list(x_test, y_test, "Testing example")

def histogram(dataset,label):
    ''' gelen datayı histogram seklinde görsellestirmeye yarar.
    Parametreleri : (dataset : gelen data , label : x etiketi  )
    kullanım : plt.hist(x, bins = number of bins) ya da
    numpy.histogram(a:input data, bins=10 : ) 

    '''

    # len(np.unique(y_train)) : number of classes
    hist,bins=np.histogram(dataset, bins=len(np.unique(y_train)))

    # plt.bar işlevi konumların ve değerlerin bir listesini alır
    # x için etiketler plt.xticks () tarafından sağlanır.

    #center = [x for x, _ in enumerate(datset)]
    
    width = 0.7 * (bins[1] - bins[0])

    center = (bins[:-1] + bins[1:]) / 2

    plt.bar(center, hist, align='center', width=width)
    plt.title("Histogram")
    plt.xlabel(label)
    plt.ylabel("Resim Sayisi")
    plt.show()



# Plotting histograms of the count of each sign

histogram(y_train, "Training examples")
histogram(y_test, "Testing examples")


# Step 3: Data Preprocessing

''' Veri önişleme için yapilacak islemler:
Shuffling , Grayscaling , Local Histogram Equalization , Normalization.

'''


















