import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

df = pd.read_csv('../VinhoClassReg.csv')

df = df.dropna(subset = ['fixed acidity', 'pH', 'volatile acidity', 'sulphates', 'citric acid'])

df['residual sugar'].mean()
df.loc[33, 'residual sugar'] = 5.445073472544469
df.loc[438, 'residual sugar'] = 5.445073472544469
df['chlorides'].mean()
df.loc[98, 'chlorides'] = 0.05605259087393658
df.loc[747, 'chlorides'] = 0.05605259087393658

def remove_outliers(df):
    df_clean = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        cutoff = std * 3
        lower, upper = mean - cutoff, mean + cutoff
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

df_clean = remove_outliers(df)

X_df = df_clean.drop('type', axis = 1)
y_df = df_clean['type']

X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(X_df, y_df, test_size = 0.3, random_state = 24)

clf_svc = SVC()

score_svc_2 = cross_val_score (clf_svc, X_df_train, y_df_train, scoring='accuracy', cv = 5, n_jobs = -1)
print("Cross-validation scores:", score_svc_2)

clf_svc_2 = clf_svc.fit(X_df_train, y_df_train)

#----------------------------------------
# save the model
import pickle

pickle.dump(clf_svc_2, open('model.pkl','wb'))

#confirm
model=pickle.load(open('model.pkl', 'rb'))
print("Predição para a entrada [7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8,6]:", model.predict([[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8,6]]))
print("Predição para a entrada [8.9,0.875,0.13,3.45,0.088,4,14,0.9994,3.44,0.52,11.5,5]:", model.predict([[8.9,0.875,0.13,3.45,0.088,4,14,0.9994,3.44,0.52,11.5,5]]))

# save the scores
pickle.dump(score_svc_2, open('score_svc.pkl','wb'))

#confirm
scores = pickle.load(open('score_svc.pkl','rb'))
print("Scores carregados:", scores)