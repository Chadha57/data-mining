import pandas as pd
from sklearn.model_selection import train_test_split #bch nkasmou test bel sk
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC 

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

train  = pd.read_csv('Train.csv')  #importation du train 
train.drop('Month',axis=1,inplace=True)   #fasakhna month
train.drop('VisitorType',axis=1,inplace=True)  #fasakhna visitortype
train.drop('Weekend',axis=1,inplace=True)    #fasakhna weekend
train = train.dropna(how="all")   # how='all' tfasakh ken les lignes li tous les valeurs te3hom manquantes
train.duplicated().sum() #habina nchoufou 3anechi haja duplique 3ata 0 donc 3anech
train = train.fillna("0") #3abina l hajet el fergha b "0"   

test  = pd.read_csv('Test.csv') # ba3ed jebna fichier test 
features = train.drop('Revenue', axis=1)
labels = train['Revenue']  #kasamneh l features w labels 

X_Train, X_Test, y_Train, y_Test = train_test_split(features,labels,test_size=0.2) #9asmet l features w labels l x_train  x_test w y_train w y_test

knn = KNeighborsClassifier()

tree = DecisionTreeClassifier()

svm = SVC()

log = LogisticRegression()

naive = GaussianNB()
#kolna l kol wehed mel model esm
knn.fit(X_Train, y_Train)

tree.fit(X_Train, y_Train)

log.fit(X_Train, y_Train)

naive.fit(X_Train, y_Train)

svm.fit(X_Train, y_Train) #fitina (X_Train, y_Train) ala kol modéle 

gbc = GradientBoostingClassifier()

rfc = RandomForestClassifier()



gbc.fit(X_Train, y_Train)

rfc.fit(X_Train, y_Train)# habina nchoufou anehou akthar wehed mouneseb

test.drop('Month',axis=1,inplace=True)
test.drop('Weekend',axis=1,inplace=True)# dropina l month el weekend wel visitortype mta3 test 
test.drop('VisitorType',axis=1,inplace=True)

for k in range(3,15):
    
  knn = KNeighborsClassifier(n_neighbors=k)

  knn.fit(X_Train, y_Train)

  y_preds = knn.predict(X_Test) # 3malna fonction bch na3mlou l y_preds bch najmou nkharjou l F1 mte3na eli bch tekhou f1_score(y_preds, y_Test, average="weighted")

for k in range(100, 150,50):
    
  rfc = RandomForestClassifier(n_estimators=k)

  rfc.fit(X_Train, y_Train)

  y_preds = rfc.predict(X_Test)# houni meme but que eli fou9ha just houni khdemna ala 150
  y_preds=rfc.predict(test)

data=pd.DataFrame(test['id'])
data['Revenue']=y_preds
data.Revenue = data['Revenue'].map( {True: 1, False: 0} )#houni hatina true:1 wel faslse:0 fel revenue bch data twali numérique w jawha behi
data.to_csv("subbmit.csv", index=False)

#bch na3mlou data w nkharajou fi westha fichier 'csv' samineh subbmit