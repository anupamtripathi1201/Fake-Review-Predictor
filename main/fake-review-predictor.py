from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re

nltk.download('stopwords')




# Text cleaning function
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords and stem words
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text


#Loading of Dataset 
df = pd.read_csv('../datasets/fake_reviews_dataset.csv')

#Cleaning of Data
df.dropna(inplace= True) #Remove rows with null or ,issing values

df.drop_duplicates(inplace= True)
df['label'] = df['label'].astype(int)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

df['cleaned_text'] = df['text'].apply(clean_text)

#PreProcessing of Text
vector = TfidfVectorizer(stop_words= 'english',max_features= 5000)

#Independant and dependant Variables

X = vector.fit_transform(df['cleaned_text'])
y = df['label']


#Train and test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25 ,random_state= 42)

#Model Setup

model = LogisticRegression()

#Hyperparamter tuning 
param_grid = {
       'C': [0.01, 0.1, 1, 10],         # Regularization strength
    'solver': ['liblinear'],         # Good for small to medium datasets
    'penalty': ['l2']                # L2 regularization

}

grid_search = GridSearchCV(model,param_grid,cv=5,n_jobs = -1,verbose = 2)
grid_search.fit(X_train,y_train)

best_model= grid_search.best_estimator_
print(grid_search.best_params_)

y_pred = best_model.predict(X_test)
print(classification_report(y_test,y_pred))


#Saving the model'
joblib.dump(best_model, 'random_forest_model.pkl')
joblib.dump(vector, 'tfidf_vectorizer.pkl')


