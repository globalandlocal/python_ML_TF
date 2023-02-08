import random

import os
import cv2
import numpy as np
import remotezip as rz
import tensorflow as tf
import einops

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'  # готовый набор с видео
files = []
# получаем  имена AVI-видео по ссылке
with rz.RemoteZip(URL) as zip:
    for zip_info in zip.infolist():
        files.append(zip_info.filename)
files = [f for f in files if f.endswith('.avi')]
# имена классов зашиты в названия видео UCF101/v_[ApplyEyeMakeup]_g01_c01.avi , извлекаем названия классов
dict_class = {}
for i in files:
    clas = i.split('_')[1]
    if clas not in dict_class:
        dict_class[clas] = []
    else:
        dict_class[clas].append(i)
clas = list(dict_class.keys())
train_video = {i: dict_class[i][:30] for i in clas}
test_video = {i: dict_class[i][30:40] for i in clas}
predict_video = {i: dict_class[i][40:50] for i in clas}
"""
with rz.RemoteZip(URL) as z:
    for i in train_video:
        x = f"./train_video/{i}/"
        for j in train_video[i]:
            z.extract(j, x)
            os.rename(f"./train_video/{i}/{j}", f"./train_video/{i}/{j.split('/')[1]}")
with rz.RemoteZip(URL) as z:
    for i in test_video:
        x = f"./test_video/{i}/"
        for j in test_video[i]:
            z.extract(j, x)
            os.rename(f"./test_video/{i}/{j}", f"./test_video/{i}/{j.split('/')[1]}")

with rz.RemoteZip(URL) as z:
    for i in predict_video:
        x = f"./predict_video/{i}/"
        for j in predict_video[i]:
            z.extract(j, x)
            os.rename(f"./predict_video/{i}/{j}", f"./predict_video/{i}/{j.split('/')[1]}")
for i in predict_video:
    os.rmdir(f"./train_video/{i}/UCF101")
    os.rmdir(f"./test_video/{i}/UCF101")
    os.rmdir(f"./predict_video/{i}/UCF101")
"""
directory = ["train_video", "test_video", "predict_video"]
train_video = {i: [j.split("/")[1] for j in train_video[i]] for i in train_video}
test_video = {i: [j.split("/")[1] for j in test_video[i]] for i in test_video}
predict_video = {i: [j.split("/")[1] for j in predict_video[i]] for i in predict_video}


def format_frame(frame, size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *size)
    return frame


def frames_per_video(path, n_frames, output_size=(250, 250), frame_step=10):
    result = []
    src = cv2.VideoCapture(path)
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step
    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)
    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    result.append(format_frame(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frame(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result

# основная проблема в том,что такое количество матриц просто не умещается в моих 16 гб оперативки,
# поэтому пришлось преобразовывать и сохранять матрицы ,а затем работать уже с ними.
# поскольку я решил сохранять принадлежность к класссам в именах матриц то все имена начинаются с 3 цифр от 000 до 101.
def save_matrix_for_video(vid, direct, num_direct, direct_save):
    c = 0
    for i in vid:
        for j in vid[i]:
            if c < 10:
                s = f"{direct_save}/00{c}{j[:-4]}.npy"
            elif c < 100:
                s = f"{direct_save}/0{c}{j[:-4]}.npy"
            else:
                s = f"{direct_save}/{c}{j[:-4]}.npy"
            np.save(s, frames_per_video(f"./{direct[num_direct]}/{i}/{j}", 10))
        c += 1
    return
# соответственно нужно создать и список путей до сохраненных матриц.

def path_to_matrix(vid):
    c = []
    label = 0
    for i in vid:
        for j in vid[i]:
            if label < 10:
                c.append(f"00{label}{j[:-4]}.npy")
            elif label < 100:
                c.append(f"0{label}{j[:-4]}.npy")
            else:
                c.append(f"{label}{j[:-4]}.npy")
        label += 1
    return c


"""
# если вы запускаете этот скрипт в первый раз уберите кавычки
# данный кусок создает соответствующие папки и превращает видео в матрицы
os.mkdir("train_matrix")
os.mkdir("test_matrix")
os.mkdir("predict_matrix")
save_matrix_for_video(train_video, directory, 0, "train_matrix")
save_matrix_for_video(test_video, directory, 1, "test_matrix")
save_matrix_for_video(predict_video, directory, 2, "predict_matrix")
"""
#  создаем список путей до матриц, перемешиваем тренировочные.
train_matrix = path_to_matrix(train_video)
random.shuffle(train_matrix)
test_matrix = path_to_matrix(test_video)
predict_matrix = path_to_matrix(predict_video)

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))
# как я уже сказал моей оперативки не хватает что бы подгрузить все матрицы разом, потому пришлось идти таким путем)
train_ds = tf.data.Dataset.from_generator(lambda: ((np.load("train_matrix/" + i), i[:3]) for i in train_matrix),
                                          output_signature=output_signature)
test_ds = tf.data.Dataset.from_generator(lambda: ((np.load("test_matrix/" + i), i[:3]) for i in test_matrix),
                                         output_signature=output_signature)
predict_ds = tf.data.Dataset.from_generator(lambda: ((np.load("predict_matrix/" + i), i[:3]) for i in predict_matrix),
                                            output_signature=output_signature)
AUTOTUNE = tf.data.AUTOTUNE
# создание кэша видео для ускорения обучения(вместо того что бы подгружать данные каждый раз только  при использовании)
# очень сильно нагружает оперативку.
train_ds = train_ds.cache().shuffle(400).prefetch(buffer_size=AUTOTUNE)
train_ds = train_ds.batch(2)
test_ds = test_ds.batch(2)
predict_ds = predict_ds.batch(2)

#большая часть конечно честно скопипастчена с уроков по тензорфлоу, я в основном поигрался с гиперпараметрами.
# класс добавляет два слоя обработки изображения один работает с шириной и высотой кадров(пространство),
# второй с временем(батчем) кадров , кратко гооря выводит результат обработки кадров(2 или больше ,
# смотря какой размер задать)
class Conv2Plus1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension.
    """
        super().__init__()
        self.seq = tf.keras.Sequential([
            # Spatial decomposition
            tf.keras.layers.Conv3D(filters=filters,
                                   kernel_size=(1, kernel_size[1], kernel_size[2]),
                                   padding=padding),
            # Temporal decomposition
            tf.keras.layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x):
        return self.seq(x)

#  приводит в должный порядок предыдущий класс, распологя между его слоями нормализацию и функцию активации
class ResidualMain(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = tf.keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            tf.keras.layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

#  данный класс используется для изменения фильтров,в случае не совпадения размерности.(используется в начале и между эпохами.
class Project(tf.keras.layers.Layer):
    """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
  """

    def __init__(self, units):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units),
            tf.keras.layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

# обьединяет работу всех предыдущим классов.
def add_residual_block(input, filters, kernel_size):
    """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
    out = ResidualMain(filters,
                       kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return tf.keras.layers.add([res, out])

#  изменения размерности изображения(как и рекомендуется большинством людей занимающихся обработкой видео нейросетями)
#  понижаем размерность, увеличиваем количество нейронов.
class ResizeVideo(tf.keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = tf.keras.layers.Resizing(self.height, self.width)

    def call(self, video):
        """
      Use the einops library to resize the tensor.

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos

#  задаем ширину и высоту того,что нейросеть будет ожидать на входе
HEIGHT = 250
WIDTH =250
input_shape = (None, 10, HEIGHT, WIDTH, 3)
# в целом я пытался создать единую модель без подобных передач х подряд, однако есть проблема при уменьшении изображения
#  сложно в Sequential передать измененные кадры в следующие слои,поэтому решил оставить как есть.
# честно предупреждаю - работать будет минимум день,
# так как в кеше всего 400 тренировочных видео(из 3030))+ 1010 тестовых видео
# по подсчетам нужно около 120гб оперативки для создания полноценного кеша
# в целом эта модель(точность около 60-70%) показывает себя хуже чем EfficientNetB0(70-80%) и обучается дольше.
input = tf.keras.layers.Input(shape=(input_shape[1:]))
x = input
x = Conv2Plus1D(filters=5, kernel_size=(3, 5, 5), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = tf.keras.layers.GlobalAveragePooling3D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(101)(x)

model = tf.keras.Model(input, x)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          epochs=50,
          validation_data=test_ds,
          callbacks=tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'))
model.evaluate(predict_ds)
