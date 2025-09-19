import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

train_data_dir = r'C:\Users\fadhlurrohman\Documents\compvis\Cats_vs_Dogs\Data\cats_dogs_light\train2'
test_data_dir = r'C:\Users\fadhlurrohman\Documents\compvis\Cats_vs_Dogs\Data\cats_dogs_light\test2'

# ====== TRAIN DATA ======
train_data = []
train_labels = []

for img in os.listdir(train_data_dir):
    img_path = os.path.join(train_data_dir, img)
    try:
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            print(f"Skip: {img_path}")
            continue
        resized_img = cv2.resize(img_array, (64, 64))
        train_data.append(resized_img.flatten())

        base_name = os.path.splitext(img)[0]
        label_str = base_name.split(' ')[0]
        label = 0 if label_str.lower() == 'cat' else 1
        train_labels.append(label)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# encode labels
label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(train_labels)

# train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(train_data, train_labels_enc)

# ====== TEST DATA ======
test_data = []
test_labels = []

for img in os.listdir(test_data_dir):
    img_path = os.path.join(test_data_dir, img)
    try:
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            print(f"Skip: {img_path}")
            continue
        resized_img = cv2.resize(img_array, (64, 64))
        test_data.append(resized_img.flatten())

        # ambil label ground truth dari nama file juga
        base_name = os.path.splitext(img)[0]
        label_str = base_name.split(' ')[0]
        label = 0 if label_str.lower() == 'cat' else 1
        test_labels.append(label)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

test_data = np.array(test_data)
test_labels = np.array(test_labels)

# prediksi
y_pred_enc = svm_model.predict(test_data)

# decode kembali ke label asli (0=cat,1=dog)
y_pred = label_encoder.inverse_transform(y_pred_enc)

# tampilkan hasil per file
for i, img in enumerate(os.listdir(test_data_dir)):
    print(f"Image {img} predicted as: {'cat' if y_pred[i]==0 else 'dog'}")

# hitung akurasi dengan label angka
acc = accuracy_score(test_labels, y_pred_enc)
print("Accuracy:", acc*100, "%")


output
(testimoni) C:\Users\fadhlurrohman\Documents\compvis\Cats_vs_Dogs\Data>C:/Users/fadhlurrohman/.conda/envs/testimoni/python.exe c:/Users/fadhlurrohman/Documents/compvis/Cats_vs_Dogs/Data/CATS_DOG/environment.py
Corrupt JPEG data: 399 extraneous bytes before marker 0xd9
Image cat (12337).jpg predicted as: dog
Image cat (12338).jpg predicted as: cat
Image cat (12339).jpg predicted as: dog
Image cat (12340).jpg predicted as: cat
Image cat (12341).jpg predicted as: dog
Image cat (12342).jpg predicted as: cat
Image cat (12343).jpg predicted as: dog
Image cat (12344).jpg predicted as: dog
Image cat (12345).jpg predicted as: cat
Image cat (12346).jpg predicted as: cat
Image cat (12347).jpg predicted as: cat
Image cat (12348).jpg predicted as: cat
Image cat (12349).jpg predicted as: dog
Image cat (12350).jpg predicted as: cat
Image cat (12351).jpg predicted as: dog
Image cat (12352).jpg predicted as: cat
Image cat.99.jpg predicted as: cat
Image dog (100).jpg predicted as: dog
Image dog (101).jpg predicted as: dog
Image dog (102).jpg predicted as: dog
Image dog (103).jpg predicted as: dog
Image dog (104).jpg predicted as: dog
Image dog (105).jpg predicted as: dog
Image dog (106).jpg predicted as: dog
Image dog (107).jpg predicted as: dog
Image dog (108).jpg predicted as: dog
Image dog (109).jpg predicted as: dog
Image dog (110).jpg predicted as: dog
Image dog (111).jpg predicted as: dog
Image dog (112).jpg predicted as: dog
Image dog (113).jpg predicted as: dog
Image dog (114).jpg predicted as: dog
Image dog (115).jpg predicted as: dog
Image dog (116).jpg predicted as: dog
Image dog (117).jpg predicted as: dog
Image dog (70).jpg predicted as: dog
Image dog (71).jpg predicted as: dog
Image dog (72).jpg predicted as: dog
Image dog (73).jpg predicted as: dog
Image dog (74).jpg predicted as: dog
Image dog (75).jpg predicted as: dog
Image dog (76).jpg predicted as: dog
Image dog (77).jpg predicted as: dog
Image dog (78).jpg predicted as: dog
Image dog (79).jpg predicted as: dog
Image dog (80).jpg predicted as: dog
Image dog (81).jpg predicted as: dog
Image dog (82).jpg predicted as: dog
Image dog (83).jpg predicted as: dog
Image dog (84).jpg predicted as: dog
Image dog (85).jpg predicted as: dog
Image dog (86).jpg predicted as: dog
Image dog (87).jpg predicted as: dog
Image dog (88).jpg predicted as: dog
Image dog (89).jpg predicted as: dog
Image dog (90).jpg predicted as: dog
Image dog (91).jpg predicted as: dog
Image dog (92).jpg predicted as: dog
Image dog (93).jpg predicted as: dog
Image dog (94).jpg predicted as: dog
Image dog (95).jpg predicted as: dog
Image dog (96).jpg predicted as: dog
Image dog (97).jpg predicted as: dog
Image dog (98).jpg predicted as: dog
Image dog (99).jpg predicted as: dog
Image dog.9807.jpg predicted as: dog
Image dog.9819.jpg predicted as: cat
Image dog.9909.jpg predicted as: cat
Image dog.9926.jpg predicted as: dog
Accuracy: 85.5072463768116 %
