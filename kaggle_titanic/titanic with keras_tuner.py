import tensorflow as tf
import numpy as np
import keras_tuner as kt

train_d = []
train_s = []
test_d = []
test_s = []
d = {'male': 1, 'female': 0}
with open('train.csv', encoding="utf-8") as g:
    print(g.readline())
    for i in g:
        x = i.split(",")
        x = [x[1], x[2], d[x[5]], x[6]]
        x = [float(i) if i != '' else .0 for i in x]
        train_s.append(x.pop(0))
        train_d.append(x)
with open('test.csv', encoding="utf-8") as g:
    print(g.readline())
    for i in g:
        x = i.split(",")
        x = [x[1], d[x[4]], x[5]]
        x = [float(i) if i != '' else .0 for i in x]
        test_d.append(x)
with open('gender_submission.csv', encoding='utf-8') as g:
    g.readline()
    for i in g:
        x = i.split(',')
        test_s.append(float(x[1]))
test_d = np.array(test_d)
test_s = np.array(test_s)
train_s = np.array(train_s)
train_d = np.array(train_d)

'''
создание модели.параметр hp передается самим тюнером и не задается учителем.
'''
def builder(hp):
    hp_unit = hp.Int("units", min_value=3, max_value=60, step=5)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])  # 0.01,0.001,0.0001
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(3),
        tf.keras.layers.Dense(hp_unit,
                              activation='relu'),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate), metrics="acc")
    return model

# собственно сам тюнер ,пока то просто создание.
tuner = kt.Hyperband(builder, objective='acc', max_epochs=10, factor=5, directory="titanic1",
                     project_name='random_titanic')
#  обратный вызов для преждевременной остановки если лосс начиначет расти,а не падать.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)
#  запуск тюнера, он будет создавать и сохранять модели в /titanic1/random_titanic
tuner.search(train_d, train_s, epochs=50, callbacks=[stop_early])
# get_best_hyperparameters() возвращает отсортироанный  реверсивно список(хотя через принт выводится объект класса),
# в котором параметры модели идут по ухудшению "score"
# однако при использовании в иницилизации модели дальше выбрасывает ошибку :
# AttributeError: 'list' object has no attribute 'Int'
best = tuner.get_best_hyperparameters(num_trials=1)[0]
# после выбора гиперпараметров для самй нейросети осталось подобрать колличество эпох для тренировки.
model = tuner.hypermodel.build(best)
history = model.fit(train_d, train_s, epochs=8, verbose=1)
val_acc = history.history['acc']
# выбираем оптимальное количество эпох и создаем окончательную модель
best_epoch = val_acc.index(max(val_acc))
hypermodel = tuner.hypermodel.build(best)
hypermodel.fit(train_d, train_s, epochs=best_epoch, verbose=1)
eval = hypermodel.evaluate(test_d,test_s)
print(eval)    #eval = [loss, acc]
'''
if eval[1]== 1:   
    model.save("./best_model.h5")
'''