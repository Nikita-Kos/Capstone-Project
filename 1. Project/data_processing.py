#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Импортируем модули и сам файл с данными о домах

# In[22]:


df = pd.read_csv("Ames_Housing_Data.csv")
df.head()


# ## Смотрим в каких колонках отсутствуют значения

# In[23]:


df.info()


# ## Смотрим на распределение данных на различных графиках

# In[24]:


sns.displot(df["SalePrice"], kde=True)


# In[25]:


sns.scatterplot(x='Overall Qual',y='SalePrice',data=df)


# ## Замечаем, что существуют в данных выбросы, которые не укладываются в общий тренд

# In[26]:


sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)


# ## Находим необходимые данные и удаляем их

# In[27]:


ind_drop = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)].index
df.drop(ind_drop, axis=0, inplace=True)
df


# ## Для дальнейшей обработки данных необходимо подгрузить файл с описанием каждой колонки

# In[28]:


with open('Ames_Housing_Feature_Description.txt','r') as f: 
    print(f.read())


# ## Данные
# 

# In[29]:


df.head()


# In[30]:


df.info()


# ### Удаление колонки PID
# 
# У нас уже есть индекс, поэтому для работы регрессии нам не нужен уникальный идентификатор PID.

# In[31]:


df = df.drop('PID',axis=1)


# ## Признаки со значениями NaN

# In[32]:


df.isnull()


# In[33]:


def percent_missing(df):
    percent_nan = 100* df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan>0].sort_values()
    return percent_nan


# In[34]:


percent_nan = percent_missing(df)


# In[35]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


# # Удаление признаков или удаление строк
# 
# Если значения отсутствуют только в нескольких строках, количество которых мало по сравнению с общим количеством строк, то можно рассмотреть вариант удалить такие строки. К чему это приведёт с точки зрения точности работы модели? По сути, мы удаляем некоторые данные для обучения и тестирования, но поскольку таких строк очень мало, то скорее всего мы не сильно повлияем на точность модели.
# 
# Если же значения отсутствуют почти во всех строках, то имеет смысл полностью удалить такие признаки. Однако перед этим следует внимательно разобраться, почему неопределённых значений так много. В некоторых случаях можно рассмотреть такие данные как отдельную категорию, отдельно от остальных данных. 

# ## Находим колонки, в которых пропущенных значений больше 1% от всех данных

# In[36]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);

# Устанавливаем пороговое значение 1% 
plt.ylim(0,1)


# Далее на основе этих данных мы или удалим строки, или заменим отсутствующие данные некоторыми значениями. 

# In[37]:


# Сравниваем с пороговым значением
percent_nan[percent_nan < 1]


# In[38]:


df[df['Total Bsmt SF'].isnull()]


# In[39]:


df[df['Bsmt Half Bath'].isnull()]


# **Заполняем данные на основе названий колонок. Здесь у нас есть два типа 2 признаков - числовые признаки и текстовые описания.**
# 
# Числовые колонки:

# In[40]:


bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)


# Текстовые колонки:

# In[41]:


bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')


# In[42]:


percent_nan = percent_missing(df)


# In[43]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


# ## Удаление строк
# 
# Некоторые признаки имеют только одну-две строки с отсутствующими значениями. На основе файла .txt с описанием признаков нашего набора данных, мы могли бы легко заполнить эти данные некоторыми значениями, и это было бы отличное решение. 

# In[44]:


df = df.dropna(axis=0,subset= ['Electrical','Garage Cars'])


# In[45]:


percent_nan = percent_missing(df)


# In[46]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);
plt.ylim(0,1)


# ### Признак "Mas Vnr"

# На основе текстового описания набора данных, отсутствие данных в признаках Mas Vnr Type и Mas Vnr Area скорее всего означает, что дом не имеет облицовки каменной плиткой, и в этом случае мы укажем нулевое значение, как мы делали раньше для других признаков.

# In[47]:


df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)


# In[48]:


percent_nan = percent_missing(df)


# In[49]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


# # Работа с отсутствующими данными в колонках
# 
# Ранее мы смотрели на отсутствие данных в строках; теперь посмотрим на колонки признаков, поскольку здесь есть достаточно большой процент отсутствующих значений.

# ### Колонки Garage
# 
# Судя по описанию данных, значение NaN означает отсутствие гаража, так что мы запишем значение "None" или 0.

# In[50]:


df[['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']]


# In[51]:


gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None')


# In[52]:


df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)


# In[53]:


percent_nan = percent_missing(df)


# In[54]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


# ## Удаление колонок с признаками
# 
# Если значения отсутствуют в достаточно большом количестве строк, то имеет смысл удалить такие колонки полностью. Например, если 99% строк имеют неопределённое значение в каком-то признаке, то этот признак не сможет использоваться для предсказывания целевой переменной, поскольку почти все данные в этом признаке неопределены. В нашем наборе данных, многие из признаков с большим количеством значений NaN по сути должны содержать значения "none" или 0. Но чтобы показать Вам различные варианты работы с отсутствующими значениями, мы удалим эти признаки вместо того, чтобы заполнить отсутствующие значения нулями или "none".

# In[55]:


percent_nan.index


# In[56]:


df[['Lot Frontage', 'Fireplace Qu', 'Fence', 'Alley', 'Misc Feature','Pool QC']]


# In[57]:


df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)


# In[58]:


percent_nan = percent_missing(df)


# In[59]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


# ### Заполняем колонку Fireplace Quality на основе текстового описания

# In[60]:


df['Fireplace Qu'] = df['Fireplace Qu'].fillna("None")


# In[61]:


percent_nan = percent_missing(df)


# In[62]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


# # Замена отсутствующих данных каким-то другим значением
# 
# Чтобы заменить отсутствующие данные в заданном признаке, нам нужно решить, какой из других признаков (без значений NaN) лучше всего коррелирует с нашим признаком. В нашем примере мы будем работать со следующими признаками:
# 
# Neighborhood: районы внутри городской черты Ames
# 
# LotFrontage: ширина фронтальной стороны дома (со стороны улицы), в футах
# 
# Мы будем работать в предположении, что признак Lot Frontage коррелирует с признаком neighborhood.

# In[63]:


df['Neighborhood'].unique()


# In[64]:


plt.figure(figsize=(8,12))
sns.boxplot(x='Lot Frontage',y='Neighborhood',data=df,orient='h')


# ## Замена отсутствующих данных на основе других признаков
# 
# Есть и более сложные методы, но обычно чем проще метод, тем лучше. Тогда мы можем не создавать модели поверх других моделей.
# 

# In[65]:


df.groupby('Neighborhood')['Lot Frontage']


# In[66]:


df.groupby('Neighborhood')['Lot Frontage'].mean()


# ## Трансформация колонки
# 

# In[67]:


df[df['Lot Frontage'].isnull()]


# In[68]:


df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))


# In[69]:


percent_nan = percent_missing(df)


# In[70]:


sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);


# In[71]:


df['Lot Frontage'] = df['Lot Frontage'].fillna(0)


# In[75]:


df.head()


# Отлично! Теперь во всём нашем наборе данных нет отсутствующих значений! 

# In[77]:


df['MS SubClass'] = df['MS SubClass'].map(str)


# In[81]:


df_obj = df.select_dtypes(include='object')
df_num = df.select_dtypes(exclude='object')


# In[88]:


df_obj = pd.get_dummies(df, drop_first=True)
final_df = pd.concat([df_num, df_obj], axis=1)
final_df

