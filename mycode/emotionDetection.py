import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import pandas as pd

angry_dir = os.path.join('./train/angry/')
disgusted_dir = os.path.join('./train/disgusted/')
fearful_dir = os.path.join('./train/fearful/')
happy_dir = os.path.join('./train/happy/')
neutral_dir = os.path.join('./train/neutral/')
sad_dir = os.path.join('./train/sad/')
surprised_dir = os.path.join('./train/surprised/')

print('total training angry images:', len(os.listdir(angry_dir)))
print('total training disgusted images:', len(os.listdir(disgusted_dir)))
print('total training fearful images:', len(os.listdir(fearful_dir)))
print('total training happy images:', len(os.listdir(happy_dir)))
print('total training neutral images:', len(os.listdir(neutral_dir)))
print('total training sad images:', len(os.listdir(sad_dir)))
print('total training surprised images:', len(os.listdir(surprised_dir)))

angry_files = os.listdir(angry_dir)
print(angry_files[:10])
disgusted_files = os.listdir(disgusted_dir)
print(disgusted_files[:10])
fearful_files = os.listdir(fearful_dir)
print(fearful_files[:10])
happy_files = os.listdir(happy_dir)
print(happy_files[:10])
neutral_files = os.listdir(neutral_dir)
print(neutral_files[:10])
sad_files = os.listdir(sad_dir)
print(sad_files[:10])
surprised_files = os.listdir(surprised_dir)
print(surprised_files[:10])

pic_index = 2

next_angry = [os.path.join(angry_dir, fname)
              for fname in angry_files[pic_index - 2: pic_index]]

for i, img_path in enumerate(next_angry):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

TRAINING_DIR = './train/'
training_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 值将在执行其他处理前乘到整个图像上
    rotation_range=40,  # 整数，数据提升时图片随机转动的角度
    width_shift_range=0.2,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
    height_shift_range=0.2,  # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度
    shear_range=0.2,  # 浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度
    zoom_range=0.2,  # 用来进行随机的放大
    validation_split=0.25,
    horizontal_flip=True,  # 布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
    fill_mode='nearest'  # 'constant','nearest','reflect','wrap'之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
)
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    subset='training',
    target_size=(48, 48),
    class_mode='categorical'
)
validation_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    subset='validation',
    target_size=(48, 48),
    class_mode='categorical'
)
print(train_generator.class_indices)

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(48, 48, 1), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=20, validation_data=validation_generator, verbose=1)

model.save('face(5).h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras.models import load_model

model = load_model("face(5).h5")

dataframe = pd.DataFrame(columns=['name', 'label'])
os.chdir('./test/')  # 原 D:\\python\\15、情绪识别
# test_num = 1
for file_name in os.listdir():
    img = image.load_img(file_name, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    num = np.argmax(classes)

    if num == 0:
        face = 'angry'
        print(
            file_name + '  ' + face)  # {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6}
    if num == 1:
        face = 'disgusted'
        print(file_name + '  ' + face)
    if num == 2:
        face = 'fearful'
        print(file_name + '  ' + face)
    if num == 3:
        face = 'happy'
        print(file_name + '  ' + face)
    if num == 4:
        face = 'neutral'
        print(file_name + '  ' + face)
    if num == 5:
        face = 'sad'
        print(file_name + '  ' + face)
    if num == 6:
        face = 'surprised'
        print(file_name + '  ' + face)
    dataframe = dataframe.append({'name': file_name, 'label': face}, ignore_index=True)
    # test_num = test_num + 1
    # if test_num > 10:
    #     break

dataframe.to_csv('submit.csv')
