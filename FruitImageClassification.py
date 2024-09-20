import sklearn as sk
from sklearn import svm
import skimage as si
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import os
import pickle
di = "train/train/"
data = []
labels = []
classes = ['Apple Braeburn','Avocado','Cherry','Banana','Kiwi','Mango','Orange','Pineapple','Strawberry','Watermelon']
for cl in classes:
    path = os.path.join(di,cl)
    for img in os.listdir(path):
        img_data = si.io.imread(os.path.join(path,img))
        img_data = si.transform.resize(img_data,(100,100,3))
        data.append(img_data.flatten())
        labels += [cl]
data = np.array(data)
labels = np.array(labels)
frame = pd.DataFrame(data)
frame['labels'] = labels
X = frame.iloc[:,:-1]
Y = frame.iloc[:,-1]
x_train,x_test,y_train,y_test=sk.model_selection.train_test_split(X,Y,test_size=0.20, random_state=77, stratify=Y) 
svc = sk.svm.SVC(probability = True)
#param_grid={'C':[0.1,1,10,100], 
            #'gamma':[0.0001,0.001,0.1,1], 
            #'kernel':['rbf','poly']}
#svc = sk.model_selection.GridSearchCV(svc,param_grid)
print("Images Processed!")
print("Starting training")
print("------------------------------")
svc.fit(x_train,y_train)
print("Done!")
y_pred = svc.predict(x_test)
y_pred_prob = svc.predict_proba(x_test)
print(y_pred_prob) 
print(svc.classes_) 
accuracy = sk.metrics.accuracy_score(y_pred, y_test) 
print(f"The model is {accuracy*100}% accurate")
pickle.dump(svc,open('model.pkl','wb'))
