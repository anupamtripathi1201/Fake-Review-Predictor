from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


#Loading of Dataset 
df = pd.read_csv('../datasets/fake_reviews_dataset.csv')

#Cleaning of Data
df.dropna(inplace= True) #Remove rows with null or ,issing values

df.drop_duplicates(inplace= True)
df['label'] = df['label'].astype(int)


#PreProcessing of Text
vector = TfidfVectorizer(stop_words= 'english',max_features= 5000)

#Independant and dependant Variables

X = vector.fit_transform(df['text'])
y = df['label']


#Train and test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25 ,random_state= 42)

#Model Setup

model = RandomForestClassifier()

#Hyperparamter tuning 
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth' : [None,10,20],
    'min_samples_split' : [2,5],
    'min_samples_leaf' : [1,2]
}

grid_search = GridSearchCV(model,param_grid,cv=5,n_jobs = -1,verbose = 2)
grid_search.fit(X_train,y_train)

best_model= grid_search.best_estimator_
print(grid_search.best_params_)

y_pred = best_model.predict(X_test)
print(classification_report(y_test,y_pred))

