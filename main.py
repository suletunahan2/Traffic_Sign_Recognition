
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

from sklearn.utils import shuffle # veri karistirma icin

'''Grayscaling(Gri Tonlama)'''
import cv2 #opencv
def  gray_scale(img):
     return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

'''image_gray=list(map(gray_scale,X_train))'''

# Plotting Gray Scale Image sample examples
# image_list(image_gray,y_train,"Gray Scale Image","gray")

'''Local Histogram Equalization (Lokal Histogram Eşitleme) 
'''
import skimage.morphology as morp #  skimage :scikit-image for Local histogram equalization
from skimage.filters import rank
def local_histogram_equ(img):
    # Local Equalization, disk shape kernel
    # Better contrast with disk kernel but could be different
    kernel=morp.disk(30)
    img_local = rank.equalize(img, selem=kernel)
    return img_local

'''
# Sample images after Local Histogram Equalization

loc_equalized_images = list(map(local_histogram_equ, image_gray))

# Plotting Sample images after Local Histogram Equalization
image_list(loc_equalized_images, y_train, "Equalized Image", "gray")'''

'''Normalization ( Normallestirme ) :       '''

def img_normalization(img):  # [0, 1] scale.
    img=np.divide(img,255)
    return img


# Sample images after normalization
'''
n = X_train.shape
normalized_images = np.zeros((n[0], n[1], n[2]))

for i, img in enumerate(loc_equalized_images):
    normalized_images[i] = img_normalization(img)

# Plotting Sample images after normalization
image_list(normalized_images, y_train, "Normalized Image", "gray")
normalized_images = normalized_images[..., None]'''


def preprocessing(data):
    ''' preprocessing function :
    Shuffling , Grayscaling , Local Histogram Equalization , Normalization.
    '''
    image_gray=list(map(gray_scale,X_train))
    loc_equalized_images = list(map(local_histogram_equ, image_gray))
    n = X_train.shape
    normalized_images = np.zeros((n[0], n[1], n[2]))

    for i, img in enumerate(loc_equalized_images):
        normalized_images[i] = img_normalization(img)

    # image_list(normalized_images, y_train, "Normalized Image", "gray")
    normalized_images = normalized_images[..., None]

    return normalized_images

X_train_preprocessed=preprocessing(X_train)
X_valid_preprocessed=preprocessing(X_valid)


X_train_preprocessed_dn = X_train_preprocessed.reshape(len(X_train_preprocessed), 32*32*1).astype('float32')
X_valid_preprocessed_dn = X_valid_preprocessed.reshape(len(X_valid_preprocessed), 32*32*1).astype('float32')

import keras
from keras.utils import to_categorical
y_train_final_dn = keras.utils.to_categorical(y_train, n_classes)
y_valid_final_dn = keras.utils.to_categorical(y_valid, n_classes)


print(X_train_preprocessed_dn.shape)
print(X_valid_preprocessed_dn.shape)
print(y_train_final_dn.shape)
print(y_valid_final_dn.shape)

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras.callbacks import ModelCheckpoint

#Building the model

class TrafficSignNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
    # CONV => RELU => BN => POOL
		model.add(Conv2D(8, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
  		# first set of (CONV => RELU => CONV => RELU) * 2 => POOL
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# second set of (CONV => RELU => CONV => RELU) * 2 => POOL
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
  		# first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		# second set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model


#data augmentation

# construct the image generator for data augmentation
from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")


# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

print("[INFO] compiling model...")

opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3,
	classes=n_classes)
model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])



# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(X_train,y_train_final_dn,batch_size=64),validation_data=(X_valid, y_valid_final_dn),
	epochs=NUM_EPOCHS)



#plotting graphs for accuracy 
plt.figure(0)
plt.plot(H.history['accuracy'], label='training accuracy')
plt.plot(H.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(H.history['loss'], label='training loss')
plt.plot(H.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


#testing accuracy on test dataset
from sklearn.metrics import accuracy_score
pred=model.predict_classes(X_test)
#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))























