# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Set image size
IMAGE_SIZE = [224, 224]

# Set paths
train_path = 'C:/test/dataset/train'
valid_path = 'C:/test/dataset/validation'

# Load VGG16 model
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze existing layers
for layer in vgg.layers:
    layer.trainable = False

# Add custom layers
folders = glob(train_path + '/*')
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# View the structure of the model
model.summary()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=10,
                                                 class_mode='categorical')

# Load validation data
test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size=(224, 224),
                                            batch_size=10,
                                            class_mode='categorical')

# Train the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=1,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save the model
model.save('chest_xray.h5')

# Load the model for prediction
model = load_model('chest_xray.h5')

# Load and preprocess a test image
img = image.load_img('C:/test/dataset/train/class1/NORMAL-170867-0002.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)

# Predict the class
classes = model.predict(img_data)
result = int(classes[0][0])

# Print the result
if result == 0:
    print("Person is Affected By PNEUMONIA")
else:
    print("Result is Normal")

# UI code
from PIL import Image, ImageEnhance
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QMovie
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(695, 609)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 701, 611))
        self.frame.setStyleSheet("background-color: #035874;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(80, -60, 541, 561))
        self.label.setText("")
        self.gif = QMovie("picture.gif")
        self.label.setMovie(self.gif)
        self.gif.start()
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(80, 430, 591, 41))
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(30, 530, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("patient.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton{\n"
                                      "border-radius: 10px;\n"
                                      " background-color:#DF582C;\n"
                                      "\n"
                                      "}\n"
                                      "QPushButton:hover {\n"
                                      " background-color: #7D93E0;\n"
                                      "}")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(450, 530, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("QPushButton{\n"
                                        "border-radius: 10px;\n"
                                        " background-color:#DF582C;\n"
                                        "\n"
                                        "}\n"
                                        "QPushButton:hover {\n"
                                        " background-color: #7D93E0;\n"
                                        "}")
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.upload_image)
        self.pushButton_2.clicked.connect(self.predict_result)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PNEUMONIA Detection Apps"))
        self.label.setToolTip(_translate("MainWindow", "<html><head/><body><p><img src=\":/newPrefix/picture.gif\"/></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Chest X_ray PNEUMONIA Detection"))
        self.pushButton.setText(_translate("MainWindow", "Upload Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Prediction"))

    def upload_image(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        path = str(path)
        print(path)
        model = load_model('chest_xray.h5')
        img_file = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img_file)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        global result
        result = classes

    def predict_result(self):
        print(result)
        if result[0][0] > 0.5:
            print("Result is Normal")
            speak("Result is Normal")
        else:
            print("Affected By PNEUMONIA")
            speak("Affected By PNEUMONIA")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())