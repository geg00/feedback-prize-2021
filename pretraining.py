import numpy as np
import pandas as pd
from sklearn import preprocessing
import config

train_df = pd.read_csv('../data/train.csv')
print (train_df['discourse_text'].head(10).values)
labels = train_df[:]['discourse_type'].unique()
le = preprocessing.LabelEncoder()
labels_num = le.fit_transform(labels)
print (labels, le.classes_, labels_num)
print (config.tokenizer(train_df['discourse_text'].values.tolist(), padding=True, truncation=True))
