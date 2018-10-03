import pandas

data_train = pandas.read_csv("/Users/yangboz/git/AI-Challenge-RTVC/mlsv2018/trainingset_annotations.txt", sep=" ", header=None)
print(data_train.head())
data_validate = pandas.read_csv("/Users/yangboz/git/AI-Challenge-RTVC/mlsv2018/validationset_annotations.txt", sep=" ", header=None)
print(data_validate.head())