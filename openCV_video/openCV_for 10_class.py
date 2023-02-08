import random
import cv2
import numpy as np
import remotezip as rz
import tensorflow as tf


URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'  # готовый набор с видео
files = []
# получаем  имена AVI-видео по ссылке
with rz.RemoteZip(URL) as zip:
    for zip_info in zip.infolist():
        files.append(zip_info.filename)
files = [f for f in files if f.endswith('.avi')]
# имена классов указаны в названия видео UCF101/v_[ApplyEyeMakeup]_g01_c01.avi , извлекаем названия классов
dict_class = {}
for i in files:
    clas = i.split('_')[1]
    if clas not in dict_class:
        dict_class[clas] = []
    else:
        dict_class[clas].append(i)
clas = list(dict_class.keys())
# извелкаем выборки ,тренировочная(30),тестовая(10),проверочная(10)
train_video = {i: dict_class[i][:30] for i in clas}
test_video = {i: dict_class[i][30:40] for i in clas}
predict_video = {i: dict_class[i][40:50] for i in clas}
"""
# при первом запуске убрать кавычки для скачивания видео и распаковки по классам.
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
# при загрузке файлы распаковываются в ****_video/UFC101/класс/ ,
# соответственно исправляем путь на ****_video/класс/название файла
directory = ["train_video", "test_video", "predict_video"]
train_video = {i: [j.split("/")[1] for j in train_video[i]] for i in train_video}
test_video = {i: [j.split("/")[1] for j in test_video[i]] for i in test_video}
predict_video = {i: [j.split("/")[1] for j in predict_video[i]] for i in predict_video}

# функции format_frame and frame_per_video честно скопипастчены из тензорфлоу уроков,
# после изучения что и как они делают решил оставить как есть, так как ничего лишнего там нет
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


# функция для извлечения путей первых 10 классов
def matrix_for_10(vid, direct, num_direct):
    mat = []
    c = 0
    for i in vid:
        for j in vid[i]:
            mat.append([frames_per_video(f"./{direct[num_direct]}/{i}/{j}", 10), c])
        c += 1
        if c == 10:
            return mat
# извлекаем пути до 10 классов и перемешиваем  выборки
# в противном случае нейросеть будет изучать по 30 видео подряд каждого класса,
# точность падает аж до 20 % в лучшем случае)


train_matrix = matrix_for_10(train_video, directory, 0)
random.shuffle(train_matrix)

test_matrix = matrix_for_10(test_video, directory, 1)
random.shuffle(test_matrix)

predict_matrix = matrix_for_10(predict_video, directory, 2)
random.shuffle(predict_matrix)
# изменение типа выходных данных из матрицы видео и меток.
output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))
#  Dataset.from_generator принимает генератор(удивительно!!!) но я решил поступить попроще,
# лямбда функция извлекает из списка путь до файла и метку класса
train_ds = tf.data.Dataset.from_generator(lambda: ((i[0], i[1]) for i in train_matrix),
                                          output_signature=output_signature)
test_ds = tf.data.Dataset.from_generator(lambda: ((i[0], i[1]) for i in test_matrix),
                                         output_signature=output_signature)
predict_ds = tf.data.Dataset.from_generator(lambda: ((i[0], i[1]) for i in predict_matrix),
                                            output_signature=output_signature)
# автотюн-автоматически подгоняет размер кэша под размер датасета,если он выше чем надо,в целом необязателен.
AUTOTUNE = tf.data.AUTOTUNE
# создание кэша видео для ускорения обучения(вместо того что бы подгружать данные каждый раз только  при использовании)
# очень сильно нагружает оперативку.
train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
predict_ds = predict_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
#  в уроке было предложено использовать заготовку готовой модели,но без загрузки весов.
#  Модель действительно хороша ,но склонна к быстрому переобучению.
net = tf.keras.applications.EfficientNetB0(include_top=False)
net.trainable = False
# к этому моменту у нас в датасете будет типичные 2д картинки.
# после ипользования функции frames_per_video мы получаем матрицу размерности :
# (10-количество кадров,250,250-высота и ширина, 3-RGB палитра)
# фактически если запускать так что это будет классификация изображений.
# поэтому нам нужно еще одно измерение -время ,которое мы получаем подавая кадры одного видео пачкой ,
# которое должно быть кратно общему количеству кадров
train_ds = train_ds.batch(5)
test_ds = test_ds.batch(5)
predict_ds = predict_ds.batch(5)
# создание собственно модели , был несколько удивлен когда при изучении и обнаружении вторым слоем Conv2D
# основной смысл TimeDistributed-наличие входного и выходного слоя ,однако она не классические возрастающая
# (когда размерность картинки за счет фильтров снижается,а количество нейронов мы наоборот увеличиваем)
# в целом TimeDistributed удобная заготовка если создавать модель не с нуля, а использовать готовые модели.
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=255),
    tf.keras.layers.TimeDistributed(net),
    tf.keras.layers.Dense(10),
    tf.keras.layers.GlobalAveragePooling3D()
])
# выбор функций оптимизации и потерь,здесь ничего особенного
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          epochs=5,
          validation_data=test_ds,
          callbacks=tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'))
# проверка предсказаний модели на данных, которых она еще не видела. точность  от 90 до 100%.
model.evaluate(predict_ds)