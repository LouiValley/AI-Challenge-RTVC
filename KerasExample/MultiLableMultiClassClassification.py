import scipy
from scipy.io import arff
import pandas as pd
data, meta = scipy.io.arff.loadarff('/Users/yangboz/PycharmProjects/KerasExample/yeast/yeast-train.arff')
df = pd.DataFrame(data)
print(df.head())
# generate dataset
from sklearn.datasets import make_multilabel_classification
X, y = make_multilabel_classification(sparse=True, n_labels= 20,
                                      return_indicator='sparse', allow_unlabeled= False)
# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
# initialize  binary relevance multi-label classifer
# with a gaussian native bayes classifer
classifer = BinaryRelevance(GaussianNB())
# generate data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# train
classifer.fit(X_train, y_train)
# predict
predictions = classifer.predict(X_test)
# calculate accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
