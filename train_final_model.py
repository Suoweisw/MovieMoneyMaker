
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


ratings=pd.read_csv('ratings.dat.txt',sep='::',engine='python',names=['UserID','MovieID','Rating','Timestamp'])
users=pd.read_csv('users.dat.txt',sep='::',engine='python',names=['UserID','TwitterID'])
movies=pd.read_csv('movies.dat.txt',sep='::',engine='python',names=['MovieID','Title','Genres'])


# In[3]:


ratings[:5]


# In[4]:


users[:5]


# In[5]:


movies[:5]


# In[7]:


df=pd.merge(pd.merge(ratings,users),movies)
df[:5]


# In[189]:


#Movie=['A Quiet Place','Suspiria','The Predator','The Nun','Slender Man','Hereditary(2018)']


# In[10]:


Movie=["A Quiet Place (2018)","Suspiria (2018)","The Predator (2018)","Overlord (2018)","The Nun (2018)",'Hereditary (2018)',"Slender Man (2018)","Ready or Not (2019)","Doctor Sleep (2019)","Midsommar (2019)","Scary Stories to Tell in the Dark (2019)","Escape Room (2019)","Child's Play (2019)"]


# In[11]:


#df0=df[df['Title'].str.contains('2018')]


# In[12]:


list=[]
for i in Movie:
    m=df.loc[df['Title']==i]
    n=m.values.tolist()
    list.append(n)
data=np.asarray(list)


# In[13]:


#list=[]
#for i in Movie:
#    m=df0[df0['Title'].str.contains(i)]
#    n=m.values.tolist()
#    list.append(n)
#data=np.asarray(list)


# In[14]:


DF=pd.DataFrame(np.concatenate(data),columns=['UserID','MovieID','Rating','Timestamp','TwitterID','Title','Genres'])
DF['Title'].unique()


# In[15]:


from datetime import datetime
import time


# In[16]:


timestamp=DF['Timestamp'].values
datetime=[]
for j in timestamp:
    z=time.strftime("%Y-%m-%d", time.localtime(int(j)))   
    datetime.append(z)
DF['Date']=datetime


# In[17]:


DF[:5]


# In[18]:


DF.Genres = DF.Genres.str.replace("'","")
DF[:5]


# In[19]:


genres=pd.DataFrame(DF.Genres.str.split('|').tolist()).stack().unique()
genres=pd.DataFrame(genres,columns=['Genre'])
genres


# In[20]:


DF=DF.join(DF.Genres.str.get_dummies().astype(bool))
DF[:5]


# In[21]:


#AQP=DF[DF['Title'].str.contains('A Quiet Place')]
AQP=DF.loc[DF['Title']=='A Quiet Place (2018)']
AQP_before=AQP[(AQP['Date'] >= '2018-03-06') & (AQP['Date'] < '2018-04-05')]
AQP_after=AQP[(AQP['Date'] >= '2018-04-06') & (AQP['Date'] < '2018-05-06')]
AQP_new=pd.concat([AQP_before,AQP_after]).sort_values(by='Date')
AQP_new.to_csv('AQP.csv',index=False)


# In[22]:


SM=DF[DF['Title'].str.contains('Slender Man')]
SM_before=SM[(SM['Date'] >= '2018-07-07') & (SM['Date'] <= '2018-08-09')]
SM_after=SM[(SM['Date'] >= '2018-08-10') & (SM['Date'] < '2018-09-10')]
SM_new=pd.concat([SM_before,SM_after]).sort_values(by='Date')


# In[23]:


Here=DF[DF['Title']=='Hereditary (2018)']
Here_before=Here[(Here['Date'] >= '2018-05-06') & (Here['Date'] < '2018-06-06')]
Here_after=Here[(Here['Date'] >= '2018-06-07') & (Here['Date'] < '2018-07-07')]
Here_new=pd.concat([Here_before,Here_after]).sort_values(by='Date')


# In[24]:


TN=DF[DF['Title'].str.contains('The Nun')]
TN_before=TN[(TN['Date'] >= '2018-08-06') & (TN['Date'] < '2018-09-06')]
TN_after=TN[(TN['Date'] >= '2018-09-07') & (TN['Date'] < '2018-10-07')]
TN_new=pd.concat([TN_before,TN_after]).sort_values(by='Date')


# In[25]:


TP=DF[DF['Title'].str.contains('The Predator')]
TP_before=TP[(TP['Date'] >= '2018-08-13') & (TP['Date'] < '2018-09-13')]
TP_after=TP[(TP['Date'] >= '2018-09-14') & (TP['Date'] < '2018-10-14')]
TP_new=pd.concat([TP_before,TP_after]).sort_values(by='Date')


# In[26]:


OL=DF[DF['Title']=='Overlord (2018)']
OL_before=OL[(OL['Date'] >= '2018-10-02') & (OL['Date'] < '2018-11-02')]
OL_after=OL[(OL['Date'] >= '2018-11-03') & (OL['Date'] < '2018-12-03')]
OL_new=pd.concat([OL_before,OL_after]).sort_values(by='Date')


# In[28]:


Sus=DF[DF['Title'].str.contains('Suspiria')]
Sus_before=Sus[(Sus['Date'] >= '2018-10-01') & (Sus['Date'] < '2018-11-01')]
Sus_after=Sus[(Sus['Date'] >= '2018-11-02') & (Sus['Date'] < '2018-12-02')]
Sus_new=pd.concat([Sus_before,Sus_after]).sort_values(by='Date')


# In[29]:


RN=DF.loc[DF['Title']=="Ready or Not (2019)"]
RN_before=RN[(RN['Date'] >= '2019-07-20') & (RN['Date'] < '2019-08-20')]
RN_after=RN[(RN['Date'] >= '2019-08-21') & (RN['Date'] < '2019-09-21')]
RN_new=pd.concat([RN_before,RN_after]).sort_values(by='Date')


# In[30]:


Mid=DF.loc[DF['Title']=="Midsommar (2019)"]
Mid_before=Mid[(Mid['Date'] >= '2019-06-02') & (Mid['Date'] < '2019-07-02')]
Mid_after=Mid[(Mid['Date'] >= '2019-07-03') & (Mid['Date'] < '2019-08-03')]
Mid_new=pd.concat([Mid_before,Mid_after]).sort_values(by='Date')


# In[31]:


SS=DF.loc[DF['Title']=="Scary Stories to Tell in the Dark (2019)"]
SS_before=SS[(SS['Date'] >= '2019-07-08') & (SS['Date'] < '2019-08-08')]
SS_after=SS[(SS['Date'] >= '2019-08-09') & (SS['Date'] < '2019-09-09')]
SS_new=pd.concat([SS_before,SS_after]).sort_values(by='Date')


# In[32]:


ER=DF.loc[DF['Title']=="Escape Room (2019)"]
ER_before=ER[(ER['Date'] >= '2018-12-03') & (ER['Date'] < '2019-01-03')]
ER_after=ER[(ER['Date'] >= '2019-01-04') & (ER['Date'] < '2019-02-04')]
ER_new=pd.concat([ER_before,ER_after]).sort_values(by='Date')


# In[33]:


DS=DF.loc[DF['Title']=="Doctor Sleep (2019)"]
DS_before=DS[(DS['Date'] >= '2019-10-07') & (DS['Date'] < '2019-11-07')]
DS_after=DS[(DS['Date'] >= '2019-11-08') & (DS['Date'] < '2019-12-08')]
DS_new=pd.concat([DS_before,DS_after]).sort_values(by='Date')


# In[34]:


#CP=DF.loc[DF['Title']=="Child's Play (2019)"]
#CP_before=CP[(CP['Date'] >= '2019-05-20') & (CP['Date'] < '2019-06-20')]
#CP_after=CP[(CP['Date'] >= '2019-06-21') & (CP['Date'] < '2019-07-21')]
#CP_new=pd.concat([DS_before,DS_after]).sort_values(by='Date')
# 1 rate


# In[59]:


frames=[AQP_new,SM_new,Here_new,TN_new,TP_new,OL_new,Sus_new,RN_new,Mid_new,SS_new,ER_new,DS_new]
movies=pd.concat(frames)
movies[:5]


# In[60]:


AQP_befmean=pd.to_numeric(AQP_before['Rating'], errors='ignore').mean()
AQP_aftmean=pd.to_numeric(AQP_after['Rating'], errors='ignore').mean()


# In[61]:


SM_befmean=pd.to_numeric(SM_before['Rating'], errors='ignore').mean()
SM_aftmean=pd.to_numeric(SM_after['Rating'], errors='ignore').mean()


# In[62]:


Here_befmean=pd.to_numeric(Here_before['Rating'], errors='ignore').mean()
Here_aftmean=pd.to_numeric(Here_after['Rating'], errors='ignore').mean()


# In[63]:


TN_befmean=pd.to_numeric(TN_before['Rating'], errors='ignore').mean()
TN_aftmean=pd.to_numeric(TN_after['Rating'], errors='ignore').mean()


# In[64]:


TP_befmean=pd.to_numeric(TP_before['Rating'],errors='ignore').mean()
TP_aftmean=pd.to_numeric(TP_after['Rating'],errors='ignore').mean()


# In[65]:


OL_befmean=pd.to_numeric(OL_before['Rating'],errors='ignore').mean()
OL_aftmean=pd.to_numeric(OL_after['Rating'],errors='ignore').mean()


# In[66]:


Sus_befmean=pd.to_numeric(Sus_before['Rating'], errors='ignore').mean()
Sus_aftmean=pd.to_numeric(Sus_after['Rating'], errors='ignore').mean()


# In[67]:


RN_befmean=pd.to_numeric(RN_before['Rating'], errors='ignore').mean()
RN_aftmean=pd.to_numeric(RN_after['Rating'], errors='ignore').mean()


# In[68]:


Mid_befmean=pd.to_numeric(Mid_before['Rating'], errors='ignore').mean()
Mid_aftmean=pd.to_numeric(Mid_after['Rating'], errors='ignore').mean()


# In[69]:


SS_befmean=pd.to_numeric(SS_before['Rating'], errors='ignore').mean()
SS_aftmean=pd.to_numeric(SS_after['Rating'], errors='ignore').mean()


# In[70]:


ER_befmean=pd.to_numeric(ER_before['Rating'], errors='ignore').mean()
ER_aftmean=pd.to_numeric(ER_after['Rating'], errors='ignore').mean()


# In[71]:


DS_befmean=pd.to_numeric(DS_before['Rating'], errors='ignore').mean()
DS_aftmean=pd.to_numeric(DS_after['Rating'], errors='ignore').mean()


# In[72]:


list0=["A Quiet Place","Slender Man","Hereditary","The Nun","The Predator","Overlord","Suspiria","Ready or Not","Midsommar","Scary Stories to Tell in the Dark","Escape Room","Doctor Sleep"]##Child's Play (2019)
list1=[AQP_befmean,SM_befmean,Here_befmean,TN_befmean,TP_befmean,OL_befmean,Sus_befmean,RN_befmean,Mid_befmean,SS_befmean,ER_befmean,DS_befmean]
list2=[AQP_aftmean,SM_aftmean,Here_aftmean,TN_aftmean,TP_aftmean,OL_aftmean,Sus_aftmean,RN_aftmean,Mid_aftmean,SS_aftmean,ER_aftmean,DS_aftmean]


# In[73]:


data={'Movie':list0,'RateBefore':list1,'RateAfter':list2}


# In[74]:


Rating = pd.DataFrame(data).fillna(0)
Rating


# In[484]:


movies[:5]


# In[485]:


movie=movies[['Title','Action','Adventure','Drama','Fantasy','Horror','Mystery','Sci-Fi','Thriller']]
movie[:5]


# In[486]:


movie[['Title','Action','Adventure','Drama','Fantasy','Horror','Mystery','Sci-Fi','Thriller']] *= 1
movie[:5]


# In[487]:


movie = movie.drop_duplicates(subset='Title').reset_index(drop=True)
movie


# In[488]:


dataset=Rating.join(movie).drop(['Title'], axis=1)
dataset


# In[489]:


dataset['Movie'].unique()


# In[490]:


dataset['RunningTime']=[91,91,127,96,98,111,153,94,138,107,99,151]


# In[491]:


dataset['Series']=[0,1,0,0,1,0,1,1,0,1,1,1]


# In[492]:


dataset['TimeBin'] = pd.cut(dataset['RunningTime'].astype(int),2)


# In[493]:


dataset


# In[494]:


dataset['Time'] = dataset['TimeBin'].astype('category').cat.codes


# In[495]:


dataset['Time'] #0 <=120 1>120


# In[496]:


dataset


# In[497]:


dataset['TimeBin'].unique()


# In[498]:


###########################################


# In[499]:


tweets_datatset=pd.read_csv('tweets_dataset.csv')
movie_pred=pd.read_csv('movie_pred.csv')


# In[500]:


full = pd.concat([tweets_datatset, movie_pred], axis=0,sort=False)
full[:5]


# In[501]:


full['Movie'].unique()


# In[502]:


full['Movie']=['Hereditary', 'Slender Man', 'A Quiet Place', 'Suspiria',
       'The Nun', 'The Predator', 'Ready or Not', 'Midsommar',
       'Scary Stories to Tell in the Dark', 'Escape Room', 'Doctor Sleep']


# In[503]:


df = pd.merge(dataset, full, on=['Movie'])


# In[504]:


df


# In[505]:


df=df.drop(['RunningTime','TimeBin'],axis=1)


# In[506]:


df[:5]


# In[507]:


df['Profitability']=[1,0,1,1,0,0,1,0,0,1,'nan']


# In[508]:


df1=df.copy()


# In[509]:


#no obvious differences between Num_of_tweets(before)	Num_of_tweets(after),Num_of_Hashtags(before)	Num_of_Hashtags(after)


# In[510]:


df1['Num_of_tweets']=(df1['Num_of_tweets(before)']+df1['Num_of_tweets(after)'])/2


# In[511]:


df1['Num_of_hashtags']=(df1['Num_of_Hashtags(before)']+df1['Num_of_Hashtags(after)'])/2


# In[512]:


#combine these 4 features to 2 features


# In[513]:


df1=df1.drop(['Num_of_tweets(before)','Num_of_tweets(after)','Num_of_Hashtags(before)','Num_of_Hashtags(after)'],axis=1)


# In[514]:


#start building models


# In[515]:


df


# In[579]:


train1=df1.loc[:5,:]


# In[580]:


test1=df1.loc[6:,:]


# In[581]:


#start training


# In[582]:


X1_train=train1.drop(['Profitability','Movie'],axis=1)
y1_train=train1['Profitability']


# In[583]:


train=df.loc[:5,:]
test=df.loc[6:,:]


# In[584]:


X_train=train.drop(['Profitability','Movie'],axis=1)
y_train=[1, 0, 1, 1, 0, 0]


# In[585]:


#random forest


# In[586]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


# In[587]:


X1_train.info()


# In[588]:


#classifier


# In[589]:


train['Profitability'].values


# In[590]:


y1_train=[1, 0, 1, 1, 0, 0]


# In[591]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


# In[592]:


model = ['logistic_regression','naive_bayes','random_forest','k_neighbors','svc']
def train_models(X_train,y_train,X_test,y_test,model):
    if model == 'logistic_regression':
        clf = LogisticRegression()
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) 
        return score
    if model == 'naive_bayes':
        clf = BernoulliNB()
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) 
        return score
    if model == 'random_forest':
        clf = RandomForestClassifier()
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) 
        return score
    if model == 'k_neighbors':
        clf = KNeighborsClassifier()
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score
    if model == 'svc':
        clf = SVC()
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) 
        return score


# In[593]:


from sklearn.model_selection import train_test_split


# In[594]:


X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_train, y_train, test_size=0.1, random_state=1)


# In[595]:


list=[]
for i in model:
    list.append(train_models(X_train_m,y_train_m,X_test_m,y_test_m,i))
models = pd.DataFrame({
'Model': model,
'Score': list})
models.sort_values(by='Score', ascending=False)


# In[596]:


clf_rf = RandomForestClassifier()      
clr_rf = clf_rf.fit(X_train,y_train)


# In[597]:


import matplotlib.pyplot as plt
importances = clr_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
plt.savefig("figure.png") # save as png


# In[598]:


clf_rf_6 = RandomForestClassifier()      
clr_rf_6 = clf_rf_6.fit(X1_train,y1_train)


# In[599]:


import matplotlib.pyplot as plt
importances = clr_rf_6.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf_6.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X1_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(X1_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X1_train.shape[1]), X1_train.columns[indices],rotation=90)
plt.xlim([-1, X1_train.shape[1]])
plt.show()
plt.savefig("figure.png") # save as png


# In[600]:


X_test=test.drop(['Profitability','Movie'],axis=1)


# In[601]:


pred=clf_rf.predict(X_test)


# In[605]:


y_test=test['Profitability']


# In[606]:


results=pd.DataFrame({'Movie':['Ready or Not','Midsommar','Scary Stories to Tell in the Dark','Escape Room','Doctor Sleep'],'Pred':pred,'test':y_test})


# In[607]:


results


# In[608]:


#


# In[609]:


X1_test=test1.drop(['Profitability','Movie'],axis=1)


# In[610]:


pred=clf_rf_6.predict(X1_test)


# In[612]:


results=pd.DataFrame({'Movie':['Ready or Not','Midsommar','Scary Stories to Tell in the Dark','Escape Room','Doctor Sleep'],'Pred':pred,'test':y_test})


# In[613]:


results

