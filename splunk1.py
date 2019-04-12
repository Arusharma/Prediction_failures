import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


df = pd.read_csv("p2.csv")
df = df.dropna(axis='columns', how='all')
#df.drop(['Activity_ID', 'Thread_ID','Win32_Thread_Id','event_id','id','RecordNumber'], axis=1, inplace=True)
df.drop(['Activity_ID', 'Thread_ID','ThreadID','_time','Win32_Thread_Id','EventId','id','RecordNumber','TimeStamp','Sid','ErrorCode'], axis=1, inplace=True)


df.loc[df.Type =='Error', 'Type'] = 0
df.loc[df.Type =='Warning', 'Type'] = 1
df.loc[df.Type =='Information', 'Type'] = 1
df.loc[(df.Type.str.contains('Exception'))==True, 'Type'] = 0
y=np.array(df['Type'])
#print(y)

limitPer = len(df) * .80
df = df.dropna(thresh=limitPer,axis=1)
print(len(df.columns.values))
df.fillna(9999,inplace=True)
df.convert_objects(convert_numeric=True)


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
#print(df.head())

X=df.loc[:, df.columns != 'Type']
y=df['Type']

'''
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feat_importances.nlargest(12))
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = ExtraTreesClassifier()
clf.fit(X_train,y_train)
clf.feature_importances_
print(clf.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
print(feat_importances.nlargest(20))
#plt.show()

print(X.shape) 

sel = SelectFromModel(clf, threshold=0.05)
sel.fit(X_train, y_train)
selected_feat= X_train.columns[(sel.get_support())]
#print(len(selected_feat))
print(selected_feat)

X_new_train = sel.transform(X_train)
X_new_test = sel.transform(X_test)

svclassifier = SVC(kernel='rbf') 
svclassifier.fit(X_new_train, y_train)
predictions = svclassifier.predict(X_new_test)  
results = confusion_matrix(y_test, predictions)
print(results)

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
print(roc_auc_score(y_test, predictions))
fpr, tpr, _ = roc_curve(y_test, predictions)
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

