# @see: https://www.depends-on-the-definition.com/classifying-genres-of-movies-by-looking-at-the-poster-a-neural-approach/
import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
#% matplotlib inline
import matplotlib.pyplot as plt

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
    plt.title("{} \n {}").format(title, genre)
show_img("114709")