import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data= pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting=3)

import re
import nltk 
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

for i in range (0,1000):
    review= re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review= review.lower()
    review= review.split()
    ps= PorterStemmer()
    allstop= stopwords.words('english')
    allstop.remove('not')
    review = [ps.stem(word) for word in review if not word in set(allstop)]
    review= ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=1500)
x= cv.fit_transform(corpus).toarray()
y= data.iloc[:,1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(xtrain,ytrain)
ypred= classifier.predict(xtest)
ypr= ypred.reshape(len(ypred),1)
ytr= ytest.reshape(len(ytest),1)
# print(np.concatenate((ypr,ytr),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(ytest,ypred)
print(cm)
print(accuracy_score(ytest,ypred))