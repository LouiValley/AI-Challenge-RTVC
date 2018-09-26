# @see: https://www.depends-on-the-definition.com/classifying-genres-of-movies-by-looking-at-the-poster-a-neural-approach/
import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
#%matplotlib inline
# @see: https://stackoverflow.com/questions/24886625/pycharm-does-not-show-plot
import matplotlib.pyplot as plt
plt.interactive(False)

# import  movie poster by genre
path = "posters/"
data = pd.read_csv("MOvieGenre.csv", encoding="ISO-8859-1")
print(data.head())
# load the movie posters
img_glob = glob.glob(path+"/"+"*.jpg")
img_dict = {}
def get_id(filename):
    index_s = filename.rfind("/") + 1
    index_f = filename.rfind(".jpg")
    return filename[index_s:index_f]
for fn in img_glob:
    try:
        img_dict[get_id(fn)] = scipy.misc.imread(fn)
    except:
        pass
def show_img(id):
    title = data[data["imdbId"] == int(id)]["Title"].values[0]
    genre = data[data["imdbId"] == int(id)]["Genre"].values[0]
    plt.imshow(img_dict[id])
    plt.title("{} \n {}".format(title, genre))
    plt.show()
show_img("19993")
# start modeling
def preprocess(img, size=(150, 101)):
    img = scipy.misc.imresize(img, size)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img
# generate dataset
def prepare_data(data, img_dict, size=(150, 101)):
    print("Generation dataset...")
    dataset = []
    y = []
    ids = []
    label_dict = {"word2idx":{},"idx2word":[]}
    idx = 0
    genre_per_movie = data["Genre"].apply(lambda x:str(x).split("|"))
    for l in [g for d in genre_per_movie for g in d]:
        if l in label_dict["idx2word"]:
            pass
        else:
            label_dict["idx2word"].append(l)
            label_dict["word2idx"][l] = idx
            idx += 1
    n_classes = len(label_dict["idx2word"])
    print("identified {} classes".format(n_classes))
    n_samples = len(img_dict)
    print("got {} samples".format(n_samples))
    for k in img_dict:
        try:
            g = data[data["imdbId"]==int(k)]["Genre"].values[0].split("|")
            img = preprocess(img_dict[k], size)
            if img.shape != (150, 101, 3):
                continue
            l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
                        for s in g], axis = 0)
            y.append(l)
            dataset.append(img)
            ids.append(k)
        except:
            pass
        print("DONE")
    return dataset, y, label_dict, ids
# scale movie poster to 96X96
SIZE = (150, 101)
dataset, y, label_dict, ids = prepare_data(data, img_dict, size=SIZE)

# build the model.