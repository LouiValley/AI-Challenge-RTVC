# example of training a final classification model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from numpy import array
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=200, verbose=0)
model.summary()
# # save model
# # serialize model to JSON
# model_json = model.to_json()
# with open('model_CRPs.json',"w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_CRPs.h5")
# # later... load model
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)
# # make a prediction
# ynew = model.predict_classes(Xnew)
# # show the inputs and predicted outputs
# for i in range(len(Xnew)):
#     print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
# Xnew = array([[0.893377589, 0.65864154]])
# # make a prediction
# ynew = model.predict_classes(Xnew)
# # show the inputs and predicted outputs
# print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
# make a prediction
ynew = model.predict_proba(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))