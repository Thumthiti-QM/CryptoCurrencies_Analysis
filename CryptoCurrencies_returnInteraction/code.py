
Stock_DJ = investpy.indices.get_index_historical_data(index = "Dow 30", country = "United States", from_date='01/01/1990',  to_date='01/01/2022')
Crypto_BTC = investpy.get_crypto_historical_data(crypto='bitcoin', from_date='01/01/1990', to_date='01/01/2022')
Alter_GOLD = investpy.get_commodity_historical_data(commodity='gold', from_date='01/01/1990', to_date='01/01/2022')

Stock_DJ['return'] = np.log(Stock_DJ['Close']) - np.log(Stock_DJ['Close']).shift(1) 
Crypto_BTC['return'] = np.log(Crypto_BTC['Close']) - np.log(Crypto_BTC['Close']).shift(1) 
Alter_GOLD['return'] = np.log(Alter_GOLD['Close']) - np.log(Alter_GOLD['Close']).shift(1) 






Alter_GOLD['date'] = Alter_GOLD.index
Alter_GOLD['year'] = Alter_GOLD.date.dt.year.astype(int) 
Alter_GOLD['par_month'] = (Alter_GOLD.date.dt.year.astype(str) + Alter_GOLD.date.dt.month.astype(str).str.zfill(2)).astype(int)
Alter_GOLD['par_week'] = (Alter_GOLD.date.dt.year.astype(str) + Alter_GOLD.date.dt.isocalendar().week.astype(str).str.zfill(2)).astype(int)


Stock_DJ['date'] = Stock_DJ.index
Stock_DJ['year'] = Stock_DJ.date.dt.year.astype(int) 
Stock_DJ['par_month'] = (Stock_DJ.date.dt.year.astype(str) + Stock_DJ.date.dt.month.astype(str).str.zfill(2)).astype(int)
Stock_DJ['par_week'] = (Stock_DJ.date.dt.year.astype(str) + Stock_DJ.date.dt.isocalendar().week.astype(str).str.zfill(2)).astype(int)

Crypto_BTC['date'] = Crypto_BTC.index
Crypto_BTC['year'] = Crypto_BTC.date.dt.year.astype(int)
Crypto_BTC['par_month'] = (Crypto_BTC.date.dt.year.astype(str) + Crypto_BTC.date.dt.month.astype(str).str.zfill(2)).astype(int)
Crypto_BTC['par_week'] = (Crypto_BTC.date.dt.year.astype(str) + Crypto_BTC.date.dt.isocalendar().week.astype(str).str.zfill(2)).astype(int)





agg_list_p = ["last"]
agg_list_r = ["std","mean"]



Crypto_BTC_grp = Crypto_BTC.groupby(['par_month']).agg({"Close":agg_list_p , "return":agg_list_r}).reset_index()


name_p = "BTC_P"
name_agg = ["par_month"]
for i in agg_list_p:
    name_agg.append(name_p + "_" + i)

name_r = "BTC_R"
for i in agg_list_r:
    name_agg.append(name_r + "_" + i)
    
    
Crypto_BTC_grp.columns = name_agg




agg_list_p = ["last"]
agg_list_r = ["std","mean"]



Alter_GOLD_grp = Alter_GOLD.groupby(['par_month']).agg({"Close":agg_list_p , "return":agg_list_r}).reset_index()


name_p = "GOLD_P"
name_agg = ["par_month"]
for i in agg_list_p:
    name_agg.append(name_p + "_" + i)

name_r = "GOLD_R"
for i in agg_list_r:
    name_agg.append(name_r + "_" + i)
    
    
Alter_GOLD_grp.columns = name_agg





agg_list_p = ["last"]
agg_list_r = ["std","mean"]



Stock_DJ_grp = Stock_DJ.groupby(['par_month']).agg({"Close":agg_list_p , "return":agg_list_r}).reset_index()


name_p = "DOW_P"
name_agg = ["par_month"]
for i in agg_list_p:
    name_agg.append(name_p + "_" + i)

name_r = "DOW_R"
for i in agg_list_r:
    name_agg.append(name_r + "_" + i)
    
    
Stock_DJ_grp.columns = name_agg






# pd.set_option("display.max_rows", 150)
prep_merge = pd.merge( Crypto_BTC_grp , 
pd.merge(Stock_DJ_grp , 
         Alter_GOLD_grp , 
         how = "left", 
         on = ["par_month"]),
         how = "left" , on = ["par_month"])

prep_merge_r = np.log(prep_merge.filter(regex='last')) - np.log(prep_merge.filter(regex='last').shift(1))

concatenated_prep = pd.concat([prep_merge[[i for i in prep_merge.columns if i not in prep_merge_r.columns]], prep_merge_r], axis=1)





### LAST return CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = concatenated_prep.filter(regex='last')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()




### vol return [on daily] CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = concatenated_prep.filter(regex='std')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()





### avg daily return on month: CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = concatenated_prep.filter(regex='mean')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()




from statsmodels.regression.linear_model import OLS

model_df= concatenated_prep.dropna()

ols_1 = OLS(model_df.filter(regex='last').filter(regex='BTC') , model_df.filter(regex='last').filter(regex='DOW'))
results_1 = ols_1.fit()



print(results_1.summary())



e_1 = model_df.filter(regex='last').filter(regex='BTC').iloc[:,0] - results_1.predict(model_df.filter(regex='last').filter(regex='DOW'))



ols_1e = OLS(e_1 , model_df.filter(regex='last').filter(regex='GOLD'))
results_1e = ols_1e.fit()


print(results_1e.summary())



ols_1e = OLS(e_1 ,   model_df.filter(regex='last').filter(regex='GOLD') )
results_1e = ols_1e.fit()




print(results_1e.summary())


# The substition effect is not strong , the compliment effect bet dow and bitcoin is sig
#check bitcoin and gold , dow and gold




ols_2 = OLS(model_df.filter(regex='last').filter(regex='BTC') , model_df.filter(regex='last').filter(regex='GOLD'))
results_2 = ols_2.fit()
print(results_2.summary())


ols_2 = OLS(model_df.filter(regex='last').filter(regex='GOLD') , model_df.filter(regex='last').filter(regex='DOW'))
results_2 = ols_2.fit()
print(results_2.summary())


ols_3 = OLS(model_df.filter(regex='last').filter(regex='BTC') , model_df.filter(regex='last')[[i for i in model_df.filter(regex='last').columns if i not in model_df.filter(regex='last').filter(regex='BTC').columns ]])
results_3 = ols_3.fit()
print(results_3.summary())


plt.plot(model_df.filter(regex='last').filter(regex='BTC'))



plt.plot(model_df.filter(regex='last').filter(regex='GOLD'))



plt.plot(model_df.filter(regex='last').filter(regex='DOW'))


model_df.filter(regex='last')


import sklearn


from sklearn.preprocessing import StandardScaler


x  = model_df.filter(regex='last')
x = StandardScaler().fit_transform(x)



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_new = pca.fit_transform(x)



fig, axes = plt.subplots(1,2)
axes[0].scatter(x[:,0], x[:,1])
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(x_new[:,0], x_new[:,1] )
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()


print(pca.explained_variance_ratio_)


colname_asset =[i for i in model_df.filter(regex='last').columns]

print(abs( pca.components_ ))



col_dat = ['BTC','DOW','GOLD']





def biplot(score, coeff , y):
    '''
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
   '''
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r','y']
    markers=['o','^','x']
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], c = colors[s], marker=markers[s]) # color based on group
    for i in range(n):
        #plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'k', alpha = 0.9,linestyle = '-',linewidth = 1.5, overhang=0.2)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, str(col_dat[i]), color = 'k', ha = 'center', va = 'center',fontsize=12)

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # reset ggplot style
# Call the biplot function for only the first 2 PCs
biplot(x_new[:,0:2], np.transpose(pca.components_[0:2, :]), y = "None")
plt.show()



agg_list_p = ["last"]
agg_list_r = ["std","mean"]



Crypto_BTC_grp = Crypto_BTC.groupby(['par_week']).agg({"Close":agg_list_p , "return":agg_list_r}).reset_index()


name_p = "BTC_P"
name_agg = ["par_week"]
for i in agg_list_p:
    name_agg.append(name_p + "_" + i)

name_r = "BTC_R"
for i in agg_list_r:
    name_agg.append(name_r + "_" + i)
    
    
Crypto_BTC_grp.columns = name_agg




agg_list_p = ["last"]
agg_list_r = ["std","mean"]



Alter_GOLD_grp = Alter_GOLD.groupby(['par_week']).agg({"Close":agg_list_p , "return":agg_list_r}).reset_index()


name_p = "GOLD_P"
name_agg = ["par_week"]
for i in agg_list_p:
    name_agg.append(name_p + "_" + i)

name_r = "GOLD_R"
for i in agg_list_r:
    name_agg.append(name_r + "_" + i)
    
    
Alter_GOLD_grp.columns = name_agg



agg_list_p = ["last"]
agg_list_r = ["std","mean"]



Stock_DJ_grp = Stock_DJ.groupby(['par_week']).agg({"Close":agg_list_p , "return":agg_list_r}).reset_index()


name_p = "DOW_P"
name_agg = ["par_week"]
for i in agg_list_p:
    name_agg.append(name_p + "_" + i)

name_r = "DOW_R"
for i in agg_list_r:
    name_agg.append(name_r + "_" + i)
    
    
Stock_DJ_grp.columns = name_agg




# pd.set_option("display.max_rows", 150)
prep_merge = pd.merge( Crypto_BTC_grp , 
pd.merge(Stock_DJ_grp , 
         Alter_GOLD_grp , 
         how = "left", 
         on = ["par_week"]),
         how = "left" , on = ["par_week"])

prep_merge_r = np.log(prep_merge.filter(regex='last')) - np.log(prep_merge.filter(regex='last').shift(1))

concatenated_prep = pd.concat([prep_merge[[i for i in prep_merge.columns if i not in prep_merge_r.columns]], prep_merge_r], axis=1)





### LAST return CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = concatenated_prep.filter(regex='last')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()


### vol return [on daily] CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = concatenated_prep.filter(regex='std')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()





### avg daily return on month: CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = concatenated_prep.filter(regex='mean')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()


from statsmodels.regression.linear_model import OLS



model_df= concatenated_prep.dropna()

ols_1 = OLS(model_df.filter(regex='last').filter(regex='BTC') , model_df.filter(regex='last').filter(regex='DOW'))
results_1 = ols_1.fit()

print(results_1.summary())






e_1 = model_df.filter(regex='last').filter(regex='BTC').iloc[:,0] - results_1.predict(model_df.filter(regex='last').filter(regex='DOW')) 



ols_1e = OLS(e_1 , model_df.filter(regex='last').filter(regex='GOLD'))
results_1e = ols_1e.fit()


print(results_1e.summary())



# Short term substituion is strong



ols_1e = OLS(  model_df.filter(regex='last').filter(regex='GOLD'),e_1   )
results_1e = ols_1e.fit()


print(results_1e.summary())







ols_2 = OLS(model_df.filter(regex='last').filter(regex='BTC') , model_df.filter(regex='last').filter(regex='GOLD'))
results_2 = ols_2.fit()
print(results_2.summary())



ols_2 = OLS(model_df.filter(regex='last').filter(regex='GOLD') , model_df.filter(regex='last').filter(regex='DOW'))
results_2 = ols_2.fit()
print(results_2.summary())





ols_3 = OLS(model_df.filter(regex='last').filter(regex='BTC') , model_df.filter(regex='last')[[i for i in model_df.filter(regex='last').columns if i not in model_df.filter(regex='last').filter(regex='BTC').columns ]])
results_3 = ols_3.fit()
print(results_3.summary())









from sklearn.preprocessing import StandardScaler


x  = model_df.filter(regex='last')
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
x_new = pca.fit_transform(x)


fig, axes = plt.subplots(1,2)
axes[0].scatter(x[:,0], x[:,1])
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(x_new[:,0], x_new[:,1] )
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()








print(pca.explained_variance_ratio_)





def biplot(score, coeff , y):
    '''
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
   '''
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r','y']
    markers=['o','^','x']
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], c = colors[s], marker=markers[s]) # color based on group
    for i in range(n):
        #plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'k', alpha = 0.9,linestyle = '-',linewidth = 1.5, overhang=0.2)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', ha = 'center', va = 'center',fontsize=10)

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # reset ggplot style
# Call the biplot function for only the first 2 PCs
biplot(x_new[:,0:2], np.transpose(pca.components_[0:2, :]), y = "None")
plt.show()






# with delete outliner




model_df_del_outline = model_df[    (abs(model_df['BTC_P_last']) 
         < abs(np.percentile(model_df['BTC_P_last'], 99))   )     
        &
         (
             abs(model_df['DOW_P_last']) 
         < abs(np.percentile(model_df['DOW_P_last'], 99))
         )
         &
         (
         abs(model_df['DOW_P_last']) 
         < abs(np.percentile(model_df['GOLD_P_last'], 99))
         )
        
        
        ]





### LAST return CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = model_df_del_outline.filter(regex='last')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()


### vol return [on daily] CORR

import seaborn as sns
import matplotlib.pyplot as plt


df_for_cor = model_df_del_outline.filter(regex='std')
correlation_mat = df_for_cor.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()






ols_2 = OLS(model_df_del_outline.filter(regex='last').filter(regex='GOLD') , model_df_del_outline.filter(regex='last').filter(regex='DOW'))
results_2 = ols_2.fit()
print(results_2.summary())






from statsmodels.regression.linear_model import OLS


model_df_del_outline= concatenated_prep.dropna()

ols_1 = OLS(model_df_del_outline.filter(regex='last').filter(regex='BTC') , model_df_del_outline.filter(regex='last').filter(regex='DOW'))
results_1 = ols_1.fit()



print(results_1.summary())





e_1 = model_df_del_outline.filter(regex='last').filter(regex='BTC').iloc[:,0] - results_1.predict(model_df_del_outline.filter(regex='last').filter(regex='DOW')) 



ols_1e = OLS(e_1 , model_df_del_outline.filter(regex='last').filter(regex='GOLD'))
results_1e = ols_1e.fit()


print(results_1e.summary())






ols_1e = OLS(e_1 ,   model_df_del_outline.filter(regex='last').filter(regex='GOLD') )
results_1e = ols_1e.fit()


print(results_1e.summary())




ols_2 = OLS(model_df_del_outline.filter(regex='last').filter(regex='BTC') , model_df_del_outline.filter(regex='last').filter(regex='GOLD'))
results_2 = ols_2.fit()
print(results_2.summary())





ols_2 = OLS(model_df_del_outline.filter(regex='last').filter(regex='GOLD') , model_df_del_outline.filter(regex='last').filter(regex='DOW'))
results_2 = ols_2.fit()
print(results_2.summary())





ols_3 = OLS(model_df_del_outline.filter(regex='last').filter(regex='BTC') , model_df_del_outline.filter(regex='last')[[i for i in model_df_del_outline.filter(regex='last').columns if i not in model_df_del_outline.filter(regex='last').filter(regex='BTC').columns ]])
results_3 = ols_3.fit()
print(results_3.summary())





model_df_del_outline.describe()







model_df_del_outline.filter(regex='last')



from sklearn.preprocessing import StandardScaler


x  = model_df_del_outline.filter(regex='last')
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_new = pca.fit_transform(x)


fig, axes = plt.subplots(1,2)
axes[0].scatter(x[:,0], x[:,1])
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(x_new[:,0], x_new[:,1] )
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()




def biplot(score, coeff , y):
    '''
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
   '''
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r','y']
    markers=['o','^','x']
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], c = colors[s], marker=markers[s]) # color based on group
    for i in range(n):
        #plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'k', alpha = 0.9,linestyle = '-',linewidth = 1.5, overhang=0.2)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', ha = 'center', va = 'center',fontsize=10)

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # reset ggplot style
# Call the biplot function for only the first 2 PCs
biplot(x_new[:,0:2], np.transpose(pca.components_[0:2, :]), y = "None")
plt.show()





# weekly with lag





model_df['BTC_P_last_l1'] = model_df['BTC_P_last'].shift(1)
model_df['BTC_P_last_l2']= model_df['BTC_P_last'].shift(2)
model_df['BTC_P_last_l3']= model_df['BTC_P_last'].shift(3)
model_df['BTC_P_last_l4']= model_df['BTC_P_last'].shift(4)


model_df['DOW_P_last_l1']= model_df['DOW_P_last'].shift(1)
model_df['DOW_P_last_l2']= model_df['DOW_P_last'].shift(2)
model_df['DOW_P_last_l3']= model_df['DOW_P_last'].shift(3)
model_df['DOW_P_last_l4']= model_df['DOW_P_last'].shift(4)

model_df['GOLD_P_last_l1']= model_df['GOLD_P_last'].shift(1)
model_df['GOLD_P_last_l2']= model_df['GOLD_P_last'].shift(2)
model_df['GOLD_P_last_l3']= model_df['GOLD_P_last'].shift(3)
model_df['GOLD_P_last_l4']= model_df['GOLD_P_last'].shift(4)






model_df2 = model_df.dropna()




# without delete outlier 




ols_3 = OLS(model_df2['BTC_P_last'] , model_df2.filter(regex='last')[[i for i in model_df2.filter(regex='last').columns if i not in model_df2.filter(regex='last').filter(regex='BTC').columns ]])
results_3 = ols_3.fit()
print(results_3.summary())




#version 1 2017 until now




model_df3 = model_df[model_df["par_week"] > 201700].dropna()


ols_3 = OLS(model_df3['BTC_P_last'] , model_df3.filter(regex='last')[[i for i in model_df3.filter(regex='last').columns if i not in model_df3.filter(regex='last').filter(regex='BTC').columns ]])
results_3 = ols_3.fit()
print(results_3.summary())








from sklearn.preprocessing import StandardScaler


x  = model_df3.filter(regex='last')[['BTC_P_last','DOW_P_last','GOLD_P_last']]
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_new = pca.fit_transform(x)


fig, axes = plt.subplots(1,2)
axes[0].scatter(x[:,0], x[:,1])
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(x_new[:,0], x_new[:,1] )
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()








def biplot(score, coeff , y):
    '''
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
   '''
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r','y']
    markers=['o','^','x']
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], c = colors[s], marker=markers[s]) # color based on group
    for i in range(n):
        #plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'k', alpha = 0.9,linestyle = '-',linewidth = 1.5, overhang=0.2)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', ha = 'center', va = 'center',fontsize=10)

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # reset ggplot style
# Call the biplot function for only the first 2 PCs
biplot(x_new[:,0:2], np.transpose(pca.components_[0:2, :]), y = "None")
plt.show()






# Without outlier



model_df3_del_outline = model_df3[    (abs(model_df3['BTC_P_last']) 
         < abs(np.percentile(model_df3['BTC_P_last'], 99))   )     
        &
         (
             abs(model_df3['DOW_P_last']) 
         < abs(np.percentile(model_df3['DOW_P_last'], 99))
         )
         &
         (
         abs(model_df3['DOW_P_last']) 
         < abs(np.percentile(model_df3['GOLD_P_last'], 99))
         )
        
        
        ]




ols_3 = OLS(model_df3_del_outline['BTC_P_last'] , model_df3_del_outline.filter(regex='last')[[i for i in model_df3_del_outline.filter(regex='last').columns if i not in model_df3_del_outline.filter(regex='last').filter(regex='BTC').columns ]])
results_3 = ols_3.fit()
print(results_3.summary())







from sklearn.preprocessing import StandardScaler


x  = model_df3_del_outline.filter(regex='last')[['BTC_P_last','DOW_P_last','GOLD_P_last']]
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_new = pca.fit_transform(x)


fig, axes = plt.subplots(1,2)
axes[0].scatter(x[:,0], x[:,1])
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(x_new[:,0], x_new[:,1] )
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()







def biplot(score, coeff , y):
    '''
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
   '''
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r','y']
    markers=['o','^','x']
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], c = colors[s], marker=markers[s]) # color based on group
    for i in range(n):
        #plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'k', alpha = 0.9,linestyle = '-',linewidth = 1.5, overhang=0.2)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', ha = 'center', va = 'center',fontsize=10)

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # reset ggplot style
# Call the biplot function for only the first 2 PCs
biplot(x_new[:,0:2], np.transpose(pca.components_[0:2, :]), y = "None")
plt.show()








