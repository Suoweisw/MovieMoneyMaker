
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('tweets_info_final.csv',encoding='latin1')


# In[3]:


df=df[['TweetId','TweetUsername','TweetText','TweetDate','Precision','Sentiment']]#around release and one month after


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


split=df.TweetDate.str.split(" ",expand=True)


# In[7]:


split=split.loc[:,0]


# In[8]:


df['Date']=split


# In[9]:


df['Date'].unique()


# In[10]:


df['Date'].replace('', np.nan, inplace=True)
df['Date'].replace('Ã¢Â\x80?Silence', np.nan, inplace=True)
df['Date'].replace('aunque', np.nan, inplace=True)
df['Date'].replace('dark', np.nan, inplace=True)
df['Date'].replace('ha', np.nan, inplace=True)


# In[11]:


#fill df's missing values and correct the date
df['Date'].fillna(method='ffill', inplace=True)  #replace na with last value
df['Date'].unique()


# In[12]:


df=df.drop(columns=['TweetDate'])


# In[13]:


a=df.loc[df['Precision'].isna()].index
df=df.drop(a,axis=0)
df = df[df['TweetText'].notnull()]


# In[14]:


#preprocessing tweet text


# In[15]:


def preprocessing(df):
    df['TweetText'] = df['TweetText'].apply(lambda x: " ".join(x.lower() for x in x.split())) #lowercase
    freq = pd.Series(' '.join(df['TweetText']).split()).value_counts()[:10]#remove most frequent words
    df['TweetText'] = df['TweetText'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    rare = pd.Series(' '.join(df['TweetText']).split()).value_counts()[-10:] #remove rare words
    df['TweetText'] = df['TweetText'].apply(lambda x: " ".join(x for x in x.split() if x not in rare))
    return df
df=preprocessing(df)


# In[16]:


#Eight movies chosen
df_h=df.loc[1:374,:]#hereditary
df_sl=df.loc[375:750,:]#slender man
df_aqp=df.loc[751:1150,:]#a quiet place
df_sus=df.loc[1189:1550,:]#suspiria
df_nun=df.loc[1551:1950,:] # the nun
df_ol = df.loc[1951:2319,:] # overlord
df_tp = df.loc[2320:2646,:] #the predator
df_m= df.loc[2647:,:] #mandy


# In[17]:


movies=[df_h,df_sl,df_aqp,df_sus,df_nun,df_ol,df_tp,df_m]
for i in movies:
    print(i['Precision'].value_counts())


# In[18]:


def relevance(df):
    relevance=len(df[df['Precision']==1])/len(df)
    return relevance


# In[19]:


p=[]
movie_name=['Hereditary','Slender Man','A Quiet Place','Suspiria','The Nun','Overlord','The predator','Mandy']
for i in movies:
    p.append(relevance(i))


# In[20]:


precision = pd.DataFrame({'Movie': movie_name,'Precision': p})
precision.sort_values(by='Precision', ascending=False)


# In[21]:


# To maintain the high quality of tweets we retrieved, drop Mandy and Overlord
movies=[df_h,df_sl,df_aqp,df_sus,df_nun,df_tp]
movie_name=['Hereditary','Slender Man','A Quiet Place','Suspiria','The Nun','The predator']


# In[22]:


#drop irrelvant tweets
def drop_irrelevant(dataframe):
    irrelevant=dataframe.loc[dataframe['Precision']==0]
    index=irrelevant.index
    dataframe=dataframe.drop(index,axis=0)
    return dataframe


# In[23]:


for idx in range(len(movies)):
    df = movies[idx]
    df=drop_irrelevant(df)


# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1),binary=True, use_idf=True)


# In[25]:


#find the sentiment need to be predicted
def predict_sentiment(df):
    df['Sentiment'] = df['Sentiment'].astype(str)
    train=df.loc[df['Sentiment'].str.isdigit()]
    pred=df.drop(train.index,axis=0)
    train['Sentiment'] = train['Sentiment'].astype(int)
    return pred,train


# In[26]:


pred_h,train_h=predict_sentiment(df_h)
pred_sl,train_sl=predict_sentiment(df_sl)
pred_aqp,train_aqp=predict_sentiment(df_aqp)
pred_sus,train_sus=predict_sentiment(df_sus)
pred_nun,train_nun=predict_sentiment(df_nun)
pred_tp,train_tp=predict_sentiment(df_tp)


# In[27]:


def drop_na(df):
    print('before',df.shape)
    print(df[df.isna().any(axis=1)].index)
    df=df.drop(df[df.isna().any(axis=1)].index,axis=0)
    print('after',df.shape)
    return df


# In[28]:


for idx in range(len(movies)):
    df = movies[idx]
    df=drop_na(df)


# In[29]:


#start training


# In[30]:


def split(train):
    X_train=train['TweetText']
    y_train=train['Sentiment']
    return X_train,y_train


# In[31]:


X_train_h,y_train_h=split(train_h)
X_pred_h,y_pred_h=split(pred_h)
X_train_sl,y_train_sl=split(train_sl)
X_pred_sl,y_pred_sl=split(pred_sl)
X_train_aqp,y_train_aqp=split(train_aqp)
X_pred_aqp,y_pred_aqp=split(pred_aqp)
X_train_sus,y_train_sus=split(train_sus)
X_pred_sus,y_pred_sus=split(pred_sus)
X_train_nun,y_train_nun=split(train_nun)
X_pred_nun,y_pred_nun=split(pred_nun)
X_train_tp,y_train_tp=split(train_tp)
X_pred_tp,y_pred_tp=split(pred_tp)


# In[32]:


def train_vect(X_train):
    train_vect = tfidf.fit_transform(X_train)
    train_vect=train_vect.toarray()
    X_train=train_vect
    return X_train


# In[33]:


X_train_h=train_vect(X_train_h)
X_pred_h=tfidf.transform(X_pred_h)
X_train_sl=train_vect(X_train_sl)
X_pred_sl=tfidf.transform(X_pred_sl)
X_train_aqp=train_vect(X_train_aqp)
X_pred_aqp=tfidf.transform(X_pred_aqp)
X_train_sus=train_vect(X_train_sus)
X_pred_sus=tfidf.transform(X_pred_sus)
X_train_tp=train_vect(X_train_tp)
X_pred_tp=tfidf.transform(X_pred_tp)


# In[34]:


#find fitted model to train


# In[35]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[36]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


# In[37]:


model = ['decision_tree','naive_bayes','linear_SGD_classifier','random_forest','k_neighbors','svc']
def train_models(X_train,y_train,X_test,y_test,model):
    if model == 'decision_tree':
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) 
        return score
    if model == 'naive_bayes':
        clf = BernoulliNB()
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) 
        return score
    if model == 'linear_SGD_classifier':
        clf = linear_model.SGDClassifier(loss="squared_loss")
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


# In[38]:


#deal with hereditary
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_train_h, y_train_h, test_size=0.2, random_state=1)


# In[39]:


list=[]
for i in model:
    list.append(train_models(Xh_train,yh_train,Xh_test,yh_test,i))
models = pd.DataFrame({
'Model': model,
'Score': list})
models.sort_values(by='Score', ascending=False)


# In[40]:


clf = RandomForestClassifier()
clf = clf.fit(X_train_h, y_train_h)
pred_h['Sentiment']=clf.predict(X_pred_h)
#combine train and pred get the whole tweets dataset
here = pd.concat([train_h, pred_h], axis=0,sort=False)
here['Sentiment']=here['Sentiment'].astype(str).astype(int)


# In[41]:


#deal with slender man
Xsl_train, Xsl_test, ysl_train, ysl_test = train_test_split(X_train_sl, y_train_sl, test_size=0.2, random_state=1)


# In[42]:


list=[]
for i in model:
    list.append(train_models(Xsl_train,ysl_train,Xsl_test,ysl_test,i))
models = pd.DataFrame({
'Model': model,
'Score': list})
models.sort_values(by='Score', ascending=False)


# In[43]:


clf = RandomForestClassifier()
clf = clf.fit(X_train_sl, y_train_sl)
pred_sl['Sentiment']=clf.predict(X_pred_sl)
#combine train and pred get the whole tweets dataset
slen = pd.concat([train_sl, pred_sl], axis=0,sort=False)
slen['Sentiment']=slen['Sentiment'].astype(str).astype(int)


# In[44]:


#deal with a quiet place
Xaqp_train, Xaqp_test, yaqp_train, yaqp_test = train_test_split(X_train_aqp, y_train_aqp, test_size=0.2, random_state=1)


# In[45]:


list=[]
for i in model:
    list.append(train_models(Xaqp_train,yaqp_train,Xaqp_test,yaqp_test,i))
models = pd.DataFrame({
'Model': model,
'Score': list})
models.sort_values(by='Score', ascending=False)


# In[46]:


clf = RandomForestClassifier()
clf = clf.fit(X_train_aqp, y_train_aqp)
pred_aqp['Sentiment']=clf.predict(X_pred_aqp)
#combine train and pred get the whole tweets dataset
aqp = pd.concat([train_aqp, pred_aqp], axis=0,sort=False)
aqp['Sentiment']=aqp['Sentiment'].astype(str).astype(int)


# In[47]:


#deal with suspiria
Xsus_train, Xsus_test, ysus_train, ysus_test = train_test_split(X_train_sus, y_train_sus, test_size=0.2, random_state=1)


# In[48]:


list=[]
for i in model:
    list.append(train_models(Xsus_train,ysus_train,Xsus_test,ysus_test,i))
models = pd.DataFrame({
'Model': model,
'Score': list})
models.sort_values(by='Score', ascending=False)


# In[49]:


clf = RandomForestClassifier()
clf = clf.fit(X_train_sus, y_train_sus)
pred_sus['Sentiment']=clf.predict(X_pred_sus)
#combine train and pred get the whole tweets dataset
sus = pd.concat([train_sus, pred_sus], axis=0,sort=False)
sus['Sentiment']=sus['Sentiment'].astype(str).astype(int)


# In[50]:


#all nun sentiment has already been labeled
nun=df_nun.drop(df_nun[df_nun['Sentiment']=='nan'].index)
nun['Sentiment']=nun['Sentiment'].astype(str).astype(int)


# In[51]:


#deal with the predator
Xtp_train, Xtp_test, ytp_train, ytp_test = train_test_split(X_train_tp, y_train_tp, test_size=0.2, random_state=1)


# In[52]:


list=[]
for i in model:
    list.append(train_models(Xtp_train,ytp_train,Xtp_test,ytp_test,i))
models = pd.DataFrame({
'Model': model,
'Score': list})
models.sort_values(by='Score', ascending=False)


# In[53]:


clf = RandomForestClassifier()
clf = clf.fit(X_train_tp, y_train_tp)
pred_tp['Sentiment']=clf.predict(X_pred_tp)
#combine train and pred get the whole tweets dataset
tp = pd.concat([train_tp, pred_tp], axis=0,sort=False)
tp['Sentiment']=tp['Sentiment'].astype(str).astype(int)


# In[54]:


def word_count(df):
    df['word_count']=df['TweetText'].apply(lambda x: len(str(x).split(" ")))
#num. of hashtags
def num_hashtags(df):
    df['hashtags']=df['TweetText'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))   
def ht(df):
    df['#']= df['TweetText'].apply(lambda x: [x for x in x.split() if x.startswith('#')])


# In[55]:


file=[here,slen,aqp,sus,nun,tp]
for i in file:
    word_count(i)
    num_hashtags(i)
    ht(i)


# In[56]:


for i in file:
    print(i['Date'].unique())


# In[57]:


#for better visualization, we divide time into for periods. We divide time into two periods(before/after) in our model.


# In[58]:


#hereditary visualization
here_1=here.loc[here.Date=='2018-06-23']#2 weeks before
here_2=here.loc[here.Date=='2018-07-06']#2 days before release
here_3=here.loc[here.Date=='2018-07-09']#2 days after releases
here_4=here.loc[here.Date=='2018-07-21']#2 weeks after release


# In[59]:


#build model using two periods
here1=here.loc[here.Date<='2018-07-06']#before 
here2=here.loc[here.Date>='2018-07-09']#after


# In[60]:


#slender man
slen_1=slen.loc[slen.Date=='2018-07-26']#2 weeks before
slen_2=slen.loc[slen.Date=='2018-08-08']#2 days before release
slen_3=slen.loc[slen.Date=='2018-08-11']#2 days after releases
slen_4=slen.loc[slen.Date=='2018-08-23']#2 weeks after release


# In[61]:


slen1=slen.loc[slen.Date<='2018-08-08']#before
slen2=slen.loc[slen.Date>'2018-08-08']#after


# In[62]:


#aqp
aqp_1=aqp.loc[aqp.Date=='2018-04-01']#2 weeks before
aqp_2=aqp.loc[aqp.Date=='2018-04-04']#2 days before release
aqp_3=aqp.loc[aqp.Date=='2018-04-06']#2 days after releases
aqp_4=aqp.loc[aqp.Date=='2018-05-05']#2 weeks after release


# In[63]:


aqp1=aqp.loc[aqp.Date<='2018-04-04']#before
aqp2=aqp.loc[aqp.Date>'2018-04-04']#after


# In[64]:


#suspiria
sus_1=sus.loc[sus.Date=='2018-10-24']#2 weeks before
sus_2=sus.loc[sus.Date=='2018-10-31']#2 days before release
sus_3=sus.loc[sus.Date=='2018-12-01']#2 days after releases
sus_4=sus.loc[sus.Date=='2018-11-02']#2 weeks after release


# In[65]:


sus1=sus.loc[sus.Date<='2018-10-31']#before
sus2=sus.loc[sus.Date>'2018-10-31']#after


# In[66]:


#the nun
nun_1=nun.loc[nun.Date=='2018-08-31']#2 weeks before
nun_2=nun.loc[nun.Date=='2018-09-05']#2 days before release
nun_3=nun.loc[nun.Date=='2018-09-06']#2 days after releases
nun_4=nun.loc[nun.Date=='2018-09-24']#2 weeks after release


# In[67]:


nun1=nun.loc[nun.Date<='2018-09-05']#before
nun2=nun.loc[nun.Date>'2018-09-05']#after


# In[68]:


#the predator
tp_1=tp.loc[tp.Date=='2018-08-28']#2 weeks before
tp_2=tp.loc[tp.Date=='2018-09-10']#2 days before release
tp_3=tp.loc[tp.Date=='2018-09-13']#2 days after releases
tp_4=tp.loc[tp.Date=='2018-09-25']#2 weeks after release


# In[69]:


tp1=tp.loc[tp.Date<='2018-09-10']#before
tp2=tp.loc[tp.Date>'2018-09-10']#after


# In[70]:


file_before=[here1,slen1,aqp1,sus1,nun1,tp1]
file_after=[here2,slen2,aqp2,sus2,nun2,tp2]


# In[71]:


sentiment_before=[]
sentiment_after=[]
num_of_hashtags_before=[]
num_of_hashtags_after=[]
num_of_tweets_before=[]
num_of_tweets_after=[]
for i in file_before:
    sentiment_before.append(i['Sentiment'].mean())
    num_of_hashtags_before.append(i['hashtags'].mean())
    num_of_tweets_before.append(len(i))
for i in file_after:
    sentiment_after.append(i['Sentiment'].mean())    
    num_of_hashtags_after.append(i['hashtags'].mean())
    num_of_tweets_after.append(len(i))


# In[74]:


dataset=pd.DataFrame({'Movie':movie_name,'Num_of_tweets(before)':num_of_tweets_before, 'Num_of_tweets(after)':num_of_tweets_after,
       'Num_of_Hashtags(before)':num_of_hashtags_before, 'Num_of_Hashtags(after)':num_of_hashtags_after,
       'Sentiment(before)':sentiment_before, 'Sentiment(after)':sentiment_after })


# In[75]:


dataset


# In[76]:


dataset.to_csv('tweets_dataset.csv', index=False)


# In[ ]:


#new movies


# In[249]:


newmovie=pd.read_csv('newmovies.csv',encoding='latin1')
newmovie=newmovie[['TweetId','TweetUsername','TweetText','TweetDate','precision','sentiment']]


# In[250]:


newmovie.columns=['TweetId','TweetUsername','TweetText','TweetDate','Precision','Sentiment']


# In[251]:


newmovie.isna().any()


# In[252]:


split2=newmovie.TweetDate.str.split(" ",expand=True)
newmovie['Date']=split2[0]


# In[253]:


newmovie['Date'].unique()


# In[254]:


newmovie['Date'].replace('', np.nan, inplace=True)


# In[255]:


#fill df's missing values and correct the date
newmovie['Date'].fillna(method='ffill', inplace=True)  #replace na with last value
newmovie['Date'].unique()


# In[269]:


ready=newmovie.loc[:198,:]
midsommar=newmovie.loc[199:388,:]
dark=newmovie.loc[389:588,:]
room=newmovie.loc[589:785,:]
doctor=newmovie.loc[786:,:]


# In[270]:


#ready or not
ready=ready.drop(ready[ready['Precision'].isnull()].index,axis=0)


# In[271]:


ready.info()


# In[272]:


ready['Precision']=ready['Precision'].astype(str).astype(int)


# In[273]:


midsommar=midsommar.drop(midsommar.loc[midsommar['Precision']=='2019-07-04 23:57:58+00:00'].index,axis=0)


# In[274]:


midsommar.info()


# In[275]:


midsommar['Precision']=midsommar['Precision'].astype(str).astype(int)


# In[276]:


dark['Precision']=dark['Precision'].astype(str).astype(int)


# In[277]:


room['Precision']=room['Precision'].astype(str).astype(int)


# In[278]:


doctor['Precision']=doctor['Precision'].astype(str).astype(int)


# In[279]:


movies=[ready,midsommar,dark,room,doctor]
movie_name=['Ready or Not','Midsommar', 'Scary Stories to Tell in the Dark', 'Escape Room','Doctor Sleep']


# In[280]:


p=[]
for i in movies:
    p.append(relevance(i))


# In[281]:


p


# In[282]:


precision = pd.DataFrame({'Movie': movie_name,'Precision': p})
precision.sort_values(by='Precision', ascending=False)


# In[283]:


#the precision of tweets we retrived about escape room is low, may influence our prediction


# In[284]:


#sentiment


# In[285]:


ready=ready.drop(ready[ready['Precision']==0].index,axis=0)
ready['Sentiment']=ready['Sentiment'].astype(int)


# In[286]:


midsommar=midsommar.drop(midsommar[midsommar['Precision']==0].index,axis=0)
midsommar['Sentiment']=midsommar['Sentiment'].astype(int)


# In[287]:


dark=dark.drop(dark[dark['Precision']==0].index,axis=0)
dark['Sentiment']=dark['Sentiment'].astype(int)


# In[288]:


room=room.drop(room[room['Precision']==0].index,axis=0)
room['Sentiment']=room['Sentiment'].astype(int)


# In[289]:


doctor=doctor.drop(doctor[doctor['Precision']==0].index,axis=0)
doctor=doctor.drop(doctor[pd.isnull(doctor).any(axis=1)].index,axis=0)
doctor['Sentiment']=doctor['Sentiment'].astype(int)


# In[294]:


file=[ready,midsommar,dark,room,doctor]
for i in file:
    word_count(i)
    num_hashtags(i)
    ht(i)


# In[231]:


#divide to two periods


# In[296]:


ready['Date'].unique()


# In[297]:


ready1=ready.loc[ready.Date<='2019-08-20']#before
ready2=ready.loc[ready.Date>'2019-08-20']#after


# In[298]:


midsommar['Date'].unique()


# In[299]:


mid1=midsommar.loc[midsommar.Date<='2019-07-02']#before
mid2=midsommar.loc[midsommar.Date>'2019-07-02']#after


# In[300]:


dark['Date'].unique()


# In[301]:


dark1=dark.loc[dark.Date<='2019-08-08']#before
dark2=dark.loc[dark.Date>'2019-08-08']#after


# In[302]:


room['Date'].unique()


# In[303]:


room1=room.loc[room.Date<='2019-01-03']#before
room2=room.loc[room.Date>'2019-01-03']#after


# In[304]:


doctor['Date'].unique()


# In[305]:


doc1=doctor.loc[doctor.Date<='2019-11-07']#before
doc2=doctor.loc[doctor.Date>'2019-11-07']#after


# In[306]:


###


# In[307]:


file_before=[ready1,mid1,dark1,room1,doc1]
file_after=[ready2,mid2,dark2,room2,doc2]


# In[312]:


sentiment_before=[]
sentiment_after=[]
num_of_hashtags_before=[]
num_of_hashtags_after=[]
num_of_tweets_before=[]
num_of_tweets_after=[]
for i in file_before:
    sentiment_before.append(i['Sentiment'].mean())
    num_of_hashtags_before.append(i['hashtags'].mean())
    num_of_tweets_before.append(len(i)*2)
for i in file_after:
    sentiment_after.append(i['Sentiment'].mean())    
    num_of_hashtags_after.append(i['hashtags'].mean())
    num_of_tweets_after.append(len(i)*2)


# In[313]:


dataset_pred=pd.DataFrame({'Movie':movie_name,'Num_of_tweets(before)':num_of_tweets_before, 'Num_of_tweets(after)':num_of_tweets_after,
       'Num_of_Hashtags(before)':num_of_hashtags_before, 'Num_of_Hashtags(after)':num_of_hashtags_after,
       'Sentiment(before)':sentiment_before, 'Sentiment(after)':sentiment_after })


# In[314]:


dataset_pred


# In[315]:


dataset_pred.to_csv('movie_pred.csv', index=False)

