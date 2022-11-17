import os
from shutil import copyfile

cat_dir = "./PetImages/Cat"
cat = os.listdir(cat_dir)
dog_dir = "./PetImages/Dog"
dog = os.listdir(dog_dir)
try:
    os.mkdir("./Cats&Dogs/")
    os.mkdir("./Cats&Dogs/Training")
    os.mkdir("./Cats&Dogs/Training/Cats")
    os.mkdir("./Cats&Dogs/Training/Dogs")
    os.mkdir("./Cats&Dogs/Testing")
    os.mkdir("./Cats&Dogs/Testing/Cats")
    os.mkdir("./Cats&Dogs/Testing/Dogs")
except FileExistsError:
    pass
x = len(cat)
x09 = x*0.9
for i in range(x):
    w = cat[i]
    if os.path.getsize(cat_dir+"/"+w) > 0:
        if i < x09:
            copyfile(cat_dir + "/" + w, "./Cats&Dogs/Training/Cats/" + w)
        else:
            copyfile(cat_dir + "/" + w, "./Cats&Dogs/Testing/Cats/" + w)
    else:
        print(cat_dir+"/"+w)
for i in range(x):
    w = dog[i]
    if os.path.getsize(dog_dir+"/"+w) > 0:
        if i < x09:
            copyfile(dog_dir + "/" + w, "./Cats&Dogs/Training/Dogs/" + w)
        else:
            copyfile(dog_dir + "/" + w, "./Cats&Dogs/Testing/Dogs/" + w)
    else:
        print(cat_dir+"/"+w)

print(len(os.listdir("./Cats&Dogs/Training/Cats")))
print(len(os.listdir("./Cats&Dogs/Training/Dogs")))
print(len(os.listdir("./Cats&Dogs/Testing/Cats")))
print(len(os.listdir("./Cats&Dogs/Testing/Dogs")))