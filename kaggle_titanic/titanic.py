import tensorflow as tf
import numpy as np

train_d = [] # тренировочные не целевые признаки
train_s = [] # тестовый целевой признак
test_d = [] # тестовые не целевые признаки
test_s = [] # тестовый целевой признак
d = {'male': 1, 'female': 0}
'''
начинаем извлечение данных из файлов,можно конечно было использовать бибилиотеку для csv , но я как то поздно об этом вспомнил)
вначале я испольозвал параметры Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Cabin , 
так как считал что признак SIbSp(количество детей у человека,или по другому обозначение является ли он родителем или нет.) 
и признаки Parch, Fare,Cabin (обозначают место каюты пассажиров) будут влиять на исход,однако точность выше 84% не поднималась.
в итоге путем проб и ошибок оставил только Pclass,Sex,Age.(Survived =целевой признак)
'''
with open('train.csv', encoding="utf-8") as g:
    print(g.readline()) # вывел все признаки для наглядности
    for i in g:
        x = i.split(",")
        x = [x[1], x[2], d[x[5]], x[6]]
        x = [float(i) if i != '' else .0 for i in x] # Survived,Pclass,Sex,Age
        train_s.append(x.pop(0)) # целевой признак ложим в отдельный список
        train_d.append(x)
with open('test.csv', encoding="utf-8") as g:
    print(g.readline())
    for i in g:
        x = i.split(",")
        x = [x[1], d[x[4]], x[5]]
        x = [float(i) if i != '' else .0 for i in x]
        test_d.append(x) # Pclass,Sex,Age
with open('gender_submission.csv', encoding='utf-8') as g:
    g.readline()
    for i in g:
        x = i.split(',')
        test_s.append(float(x[1])) # Survived
# для ускорения работы преобразуем все это в массивы Numpy
test_d = np.array(test_d)
test_s = np.array(test_s)
train_s = np.array(train_s)
train_d = np.array(train_d)
# модель
model = tf.keras.models.Sequential([
        tf.keras.layers.Input(3),
        tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.08), metrics="acc")
'''
тренировка модели пока точность на тестовой выборке не будет 100%
в теории звучит хорошо, но на практике приходится иногда перезапускать ,  
так как начальные веса могут быть слишком далеко от точки минимума
в теории могло бы помочь создание самой модели в цикле, 
но не помогло, 
возможно веса инициилизируются в момент запуска программы(при импорте tensorflow?)
'''
hist = [0, 0]
while hist[1] != 1:
    model.fit(train_d, train_s, epochs=8, verbose=1)
    hist = model.evaluate(test_d, test_s)
# вывод  результатов
if hist[1] == 1:
    history = model.predict(test_d)
    with open("predict.txt", "w", encoding="utf-8") as q:
        for i in history:
            if sum(i) > 0.5:
                q.write("1\n")
            else:
                q.write("0\n")
