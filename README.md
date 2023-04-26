<div class="cell markdown" id="TvkLyUMrjWP6">

# Проектна задача по Вовед во науката за податоци

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="_j-K17mujjRu">

## Виктор Мегленовски 191001 - ФИНКИ 2021/2022

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="HYDwB8LVjsIM">

Линк до податочно множество:
<https://www.kaggle.com/datasets/sovannt/world-bank-youth-unemployment>

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="U2JtqBbAkAZC">

## 1.Импортирање на потребни библиотеки

</div>

<div class="cell code" execution_count="306"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="5uv7RJDpA5CZ" outputId="98638732-9f55-4770-d4c4-ba9e21e7e18a">

``` python
!pip install catboost
```

<div class="output stream stdout">

    Requirement already satisfied: catboost in /usr/local/lib/python3.7/dist-packages (1.0.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from catboost) (3.2.2)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.3.5)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.7/dist-packages (from catboost) (0.10.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from catboost) (1.4.1)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from catboost) (1.15.0)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.21.6)
    Requirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (from catboost) (5.5.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2022.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (1.4.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (3.0.8)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->catboost) (4.2.0)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from plotly->catboost) (8.0.1)

</div>

</div>

<div class="cell code" execution_count="307" id="K8pNf0bWlMQN">

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, f1_score
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense,Embedding,Conv1D,MaxPooling1D,LSTM, Flatten, Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
```

</div>

<div class="cell markdown" id="J2FFW1xrmlAa">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="6IGcntL_kTFO">

## 2.Поврзување со Google Drive и вчитување на податоците

</div>

<div class="cell code" execution_count="308" id="tgn4C_O1kZHW">

``` python
original_df=pd.read_csv("/content/drive/MyDrive/API_ILO_country_YU.csv")
```

</div>

<div class="cell code" execution_count="401"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="J0PZHVfhle4x" outputId="89157d65-5b48-41d8-8a36-ce18e5dd7051">

``` python
print(original_df.head())
```

<div class="output stream stdout">

               Country Name Country Code       2010       2011       2012  \
    0           Afghanistan          AFG  20.600000  20.900000  19.700001   
    1                Angola          AGO  10.800000  10.700000  10.700000   
    2               Albania          ALB  25.799999  27.000000  28.299999   
    3            Arab World          ARB  25.022214  28.117516  29.113212   
    4  United Arab Emirates          ARE   9.800000   9.800000   9.800000   

            2013       2014  
    0  21.100000  20.799999  
    1  10.600000  10.500000  
    2  28.700001  29.200001  
    3  29.335306  29.704569  
    4   9.900000  10.000000  

</div>

</div>

<div class="cell code" execution_count="402" id="vK3nrCgAlgnu">

``` python
df=original_df.copy()
```

</div>

<div class="cell code" execution_count="403"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="S6vzN_qhllD9" outputId="c48307d7-f6fa-4f83-c3c6-c296e6275233">

``` python
print(df.head())
```

<div class="output stream stdout">

               Country Name Country Code       2010       2011       2012  \
    0           Afghanistan          AFG  20.600000  20.900000  19.700001   
    1                Angola          AGO  10.800000  10.700000  10.700000   
    2               Albania          ALB  25.799999  27.000000  28.299999   
    3            Arab World          ARB  25.022214  28.117516  29.113212   
    4  United Arab Emirates          ARE   9.800000   9.800000   9.800000   

            2013       2014  
    0  21.100000  20.799999  
    1  10.600000  10.500000  
    2  28.700001  29.200001  
    3  29.335306  29.704569  
    4   9.900000  10.000000  

</div>

</div>

<div class="cell markdown" id="Ko2swX0fmDfW">

*Ги отстрануваме првите 2 колони бидејќи не ни даваат никакво семантичко
значење во врска со проблемот.*

</div>

<div class="cell code" execution_count="404" id="YbLlmp2vllu-">

``` python
df.drop(['Country Name', 'Country Code'], axis = 1,inplace=True)
```

</div>

<div class="cell code" execution_count="405"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Wg62-wX0l7L8" outputId="c0c5dabe-1cbf-4aa5-8ceb-5e877571bcbc">

``` python
print(df.head())
```

<div class="output stream stdout">

            2010       2011       2012       2013       2014
    0  20.600000  20.900000  19.700001  21.100000  20.799999
    1  10.800000  10.700000  10.700000  10.600000  10.500000
    2  25.799999  27.000000  28.299999  28.700001  29.200001
    3  25.022214  28.117516  29.113212  29.335306  29.704569
    4   9.800000   9.800000   9.800000   9.900000  10.000000

</div>

</div>

<div class="cell code" execution_count="406"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Af3VqXrRl8og" outputId="45fa811a-fdd9-4a70-a293-8a13c1581749">

``` python
print(df.info())
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 219 entries, 0 to 218
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   2010    219 non-null    float64
     1   2011    219 non-null    float64
     2   2012    219 non-null    float64
     3   2013    219 non-null    float64
     4   2014    219 non-null    float64
    dtypes: float64(5)
    memory usage: 8.7 KB
    None

</div>

</div>

<div class="cell markdown" id="8mtvezpimm6i">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="zEz_uioSnH3C">

## 3.Дескриптивни статистики за променливите

*Сите променливи се од непрекинат тип*

</div>

<div class="cell code" execution_count="315"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="S73kppv3nSH5" outputId="98dda119-4955-4958-ab2c-d5b38339a801">

``` python
print('Mean')
print(df.mean())
```

<div class="output stream stdout">

    Mean
    2010    17.892957
    2011    17.902713
    2012    18.148142
    2013    18.100429
    2014    17.943539
    dtype: float64

</div>

</div>

<div class="cell code" execution_count="316"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Cu0cbDrPnzvM" outputId="a9dfe8ee-222c-4f5c-c744-4888cf9d3626">

``` python
print('Median')
print(df.median())
```

<div class="output stream stdout">

    Median
    2010    14.900000
    2011    14.523908
    2012    14.400000
    2013    14.100000
    2014    14.124300
    dtype: float64

</div>

</div>

<div class="cell code" execution_count="317"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="9KN9I9ReoPjP" outputId="048b9528-cdfd-4997-b1fc-2f3400e2f38f">

``` python
print('Standard deviation')
print(df.std())
```

<div class="output stream stdout">

    Standard deviation
    2010    10.540099
    2011    10.887558
    2012    11.430862
    2013    11.674366
    2014    11.554674
    dtype: float64

</div>

</div>

<div class="cell code" execution_count="318"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Kk1-vk69oSeT" outputId="570cd78f-6c61-45c3-bac9-5abfe4af5660">

``` python
print('Minimum')
print(df.min())
```

<div class="output stream stdout">

    Minimum
    2010    0.7
    2011    0.7
    2012    0.5
    2013    0.7
    2014    0.7
    dtype: float64

</div>

</div>

<div class="cell code" execution_count="319"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="3Wt0NDY5oUiK" outputId="6489e672-5309-4b5f-f3e9-06a02922bfb7">

``` python
print('Maximum')
print(df.max())
```

<div class="output stream stdout">

    Maximum
    2010    57.200001
    2011    57.099998
    2012    61.700001
    2013    58.000000
    2014    57.900002
    dtype: float64

</div>

</div>

<div class="cell code" execution_count="320"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="k4CP7px6oWxb" outputId="aceecf91-9190-442a-fe82-742b20672f8f">

``` python
print('Quantiles')
print(df.quantile([.1, .25, .5, .75], axis = 0)) 
```

<div class="output stream stdout">

    Quantiles
           2010       2011       2012       2013       2014
    0.10   7.04   7.380000   6.720000   6.540000   6.660000
    0.25  10.60  10.410530  10.500000  10.490677  10.500000
    0.50  14.90  14.523908  14.400000  14.100000  14.124300
    0.75  23.00  23.200001  24.616293  23.435561  23.310668

</div>

</div>

<div class="cell markdown" id="8wlXwnxxooI6">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="esAya_bJo_Rt">

## 4.Визуелизација на податоците

</div>

<div class="cell markdown" id="mbg5KtC2K2Gc">

### 4.1.Дистрибуција на променливите

</div>

<div class="cell code" execution_count="321"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:409}"
id="8tdbxJz6oZe6" outputId="304884f6-c94a-4a78-8bf5-e4ba1b062e84">

``` python
df.hist(bins = 5)
```

<div class="output execute_result" execution_count="321">

    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f59acb52e50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f59b77f5d10>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f59b6e18250>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f59b9482750>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f59b9438c50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f59b93fc190>]],
          dtype=object)

</div>

<div class="output display_data">

![](46d24264a6b95901c5666a4a670611f0de1dd42d.png)

</div>

</div>

<div class="cell markdown" id="ySRT9Cm8LDTq">

### 4.2.Тренд низ времето

</div>

<div class="cell code" execution_count="322"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:284}"
id="R21mdMIkpoyV" outputId="63a09452-21cf-444e-b07a-aa1cc265ca50">

``` python
df.plot.area()
```

<div class="output execute_result" execution_count="322">

    <matplotlib.axes._subplots.AxesSubplot at 0x7f59b9396910>

</div>

<div class="output display_data">

![](3184459c96cb17abbfd733c593fa6f54d48ed27b.png)

</div>

</div>

<div class="cell markdown" id="DlyHtBrxLHfB">

### 4.3.Врска помеѓу променливите

</div>

<div class="cell markdown" id="A9m1icaZLMBm">

#### 2010 и 2011

</div>

<div class="cell code" execution_count="323"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:337}"
id="-vjxCuC_KZOr" outputId="57e0ef5c-a616-4ac8-c0a6-29ba0a42c541">

``` python
fig = plt.figure()
df.plot.scatter(x= '2010', y = '2011', alpha = 0.75,rot=0)
plt.xticks(rotation=90)
```

<div class="output execute_result" execution_count="323">

    (array([-10.,   0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.]),
     <a list of 9 Text major ticklabel objects>)

</div>

<div class="output display_data">

    <Figure size 432x288 with 0 Axes>

</div>

<div class="output display_data">

![](f474eaccfbc91b9c1186e0a60bd35c01a1d755de.png)

</div>

</div>

<div class="cell markdown" id="nQfQIzWZLUsw">

#### 2011 и 2012

</div>

<div class="cell code" execution_count="324"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:337}"
id="-Dy896cpLQuO" outputId="7b80257e-291d-4d90-e338-341885974e1b">

``` python
fig = plt.figure()
df.plot.scatter(x= '2011', y = '2012', alpha = 0.75,rot=0)
plt.xticks(rotation=90)
```

<div class="output execute_result" execution_count="324">

    (array([-10.,   0.,  10.,  20.,  30.,  40.,  50.,  60.]),
     <a list of 8 Text major ticklabel objects>)

</div>

<div class="output display_data">

    <Figure size 432x288 with 0 Axes>

</div>

<div class="output display_data">

![](bfdc0e8ac84678471752a1ee56b05b2335494a6e.png)

</div>

</div>

<div class="cell markdown" id="ndKY-41ELbOK">

#### 2012 и 2013

</div>

<div class="cell code" execution_count="325"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:338}"
id="JDK9pDhwLXQO" outputId="22527842-ae75-4a59-dd61-1c2d0fcedaf4">

``` python
fig = plt.figure()
df.plot.scatter(x= '2012', y = '2013', alpha = 0.75,rot=0)
plt.xticks(rotation=90)
```

<div class="output execute_result" execution_count="325">

    (array([-10.,   0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.]),
     <a list of 9 Text major ticklabel objects>)

</div>

<div class="output display_data">

    <Figure size 432x288 with 0 Axes>

</div>

<div class="output display_data">

![](ca5bae0739df3bbeee7446962bd4ebdd178bfad2.png)

</div>

</div>

<div class="cell markdown" id="S1Ee-YDiLfJJ">

#### 2013 и 2014

</div>

<div class="cell code" execution_count="326"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:338}"
id="U_lciyDGLcRd" outputId="f5b82618-1a2c-4bda-c33e-66dcd31dda6f">

``` python
fig = plt.figure()
df.plot.scatter(x= '2013', y = '2014', alpha = 0.75,rot=0)
plt.xticks(rotation=90)
```

<div class="output execute_result" execution_count="326">

    (array([-10.,   0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.]),
     <a list of 9 Text major ticklabel objects>)

</div>

<div class="output display_data">

    <Figure size 432x288 with 0 Axes>

</div>

<div class="output display_data">

![](9d1d448725079933a11a3a413bd31de4d45a3fb0.png)

</div>

</div>

<div class="cell markdown" id="Cgwz6e8nLmnK">

*Може да заклучиме дека помеѓу сите парови од соседни години постои јака
линеарна зависност со коефициент блиску до 1*

</div>

<div class="cell code" execution_count="327"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:921}"
id="58ikCiuKNQCz" outputId="88b9798e-715c-4013-c8a6-157a35ea9ea0">

``` python
sns.pairplot(df)
```

<div class="output execute_result" execution_count="327">

    <seaborn.axisgrid.PairGrid at 0x7f59bb224290>

</div>

<div class="output display_data">

![](6c533d16990faea052510d362fab19feb5a1031c.png)

</div>

</div>

<div class="cell markdown" id="0f-Q0Q6BLyHl">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="blNABxWZMF4o">

## 5.Справување со вредности кои недостасуваат

</div>

<div class="cell markdown" id="692eu6u7MR2C">

Најпрво правиме проверка дали постојат вредности кои недостасуваат во
таргет променливата, доколку постојат такви вредности, истите записи ги
отстрануваме.

</div>

<div class="cell code" execution_count="407"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="EHwclyyHLhwS" outputId="fd705576-c707-4a95-9069-8658b705f5f7">

``` python
missing_values = df.isnull().sum()
percentage = 100 * df.isnull().sum() / len(df)
missing_values_table = pd.concat([missing_values, percentage], axis=1)
missing_values_table.columns = ['Num. of missing values','% of missing values']
print(missing_values_table)
```

<div class="output stream stdout">

          Num. of missing values  % of missing values
    2010                       0                  0.0
    2011                       0                  0.0
    2012                       0                  0.0
    2013                       0                  0.0
    2014                       0                  0.0

</div>

</div>

<div class="cell markdown" id="_S_0nAUGMo82">

Како што можеме да заклучиме, во податоците не постојат вредности кои
недостасуваат.

</div>

<div class="cell markdown" id="sAdPYcJPM6d_">

Исто така немаме категориски променливи, па нема потреба да правиме
енкодирање на истите.

</div>

<div class="cell markdown" id="6yaqLEf6MtFf">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="hk4IKG_8Mu7W">

## 6.Поделба на податоците за тренирање и за тестирање

</div>

<div class="cell code" execution_count="329" id="P6IWpysAMm1J">

``` python
X_train_original, X_test_original, Y_train_original, Y_test_original = train_test_split(df[df.columns[:-1]],  df['2014'], test_size=0.2)
```

</div>

<div class="cell markdown" id="GAjrZAAkO-98">

Правиме скалирање на вредностите

</div>

<div class="cell code" execution_count="330" id="ZJnfIguZO2zj">

``` python
scaler = StandardScaler()
scaler.fit(X_train_original)

X_train_original = scaler.transform(X_train_original)
X_test_original = scaler.transform(X_test_original)
```

</div>

<div class="cell code" execution_count="331" id="N07MLvJDhrBN">

``` python
matrix=[]
```

</div>

<div class="cell code" execution_count="332" id="LZ84j28WNtjQ">

``` python
def make_copies():
  return X_train_original.copy(), X_test_original.copy(), Y_train_original.copy(), Y_test_original.copy()
```

</div>

<div class="cell markdown" id="vC7AoTEqOcnf">

*Горната функција ќе се користи секогаш кога ќе имаме потреба да
направиме копија од податоците онака како што сме ги поделиле првиот
пат*

</div>

<div class="cell markdown" id="m8CBi5DjOoda">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="0rXW81kvVyKX">

## 7.Модел со линеарна регресија

</div>

<div class="cell code" execution_count="333" id="G9xfgHeAV9pB">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="334" id="UDm_ih1wOWzR">

``` python
model = LinearRegression().fit(X_train, Y_train)
```

</div>

<div class="cell code" execution_count="335"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="-Ogjj13sWKp0" outputId="82431c96-563d-4b25-b1ad-9262693411c3">

``` python
y_pred = model.predict(X_test)
y_pred
```

<div class="output execute_result" execution_count="335">

    array([16.53865385, 35.28083688,  5.59525541, 13.22126889,  9.79714022,
           23.51590577, 10.06735085, 13.1759868 , 30.64204975, 12.84062058,
            7.13211111,  8.20513145,  5.95386066,  9.80776417, 30.55514694,
           13.80610334,  1.48400931, 28.30065383, 23.40672013, 18.14601188,
           28.49265619, 19.52144076, 26.67706307, 10.40629802, 13.96456371,
            9.33038415, 35.23636052, 19.46817511, 32.80406134, 14.09424033,
           11.14349099, 16.44067166, 13.35699666, 10.59319382,  4.15890669,
            1.2160964 , 41.63318911, 40.55985646,  7.80358369, 22.09691099,
            9.21670467, 18.80343432, 10.93321278, 10.5112536 ])

</div>

</div>

<div class="cell code" execution_count="336"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="TfiMwjp7oUOh" outputId="ce0df04b-808b-428c-e21b-21e9fb47ddb6">

``` python
print('R^2 Score: ', r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 Score:  0.9910553470777943

</div>

</div>

<div class="cell code" execution_count="337"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Mwtg9XcNWg_C" outputId="ee833158-cc4a-45fd-d332-e608386aa0d2">

``` python
print("Mean squared error: ",mean_squared_error(Y_test,y_pred))
```

<div class="output stream stdout">

    Mean squared error:  0.9565766251911539

</div>

</div>

<div class="cell code" execution_count="338" id="LZ70LHWahtRJ">

``` python
matrix.append(["Линеарна регресија",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="s7KbEEIgYfJq">

*Овој модел дава одлични резултати*

</div>

<div class="cell markdown" id="h3C7omoaXN1m">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="7REs8T_IXkio">

## 8.Модел со полиномна регресија

</div>

<div class="cell markdown" id="sRTyfEdmYumM">

### 8.1.Полиномна регресија со степен 4

</div>

<div class="cell code" execution_count="339" id="KS2UEskRY6pP">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="340" id="mThyhHeQW6vQ">

``` python
poly_reg=PolynomialFeatures(4)
X_train_poly=poly_reg.fit_transform(X_train)
poly_reg.fit(X_train_poly,Y_train)

lin_reg=LinearRegression()
lin_reg.fit(X_train_poly,Y_train)

X_test_poly=poly_reg.fit_transform(X_test)
y_pred=lin_reg.predict(X_test_poly)

training_score=mean_squared_error(Y_train, lin_reg.predict(X_train_poly))
test_score=mean_squared_error(Y_test,y_pred)
```

</div>

<div class="cell code" execution_count="341"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="OyhFzA5yoO0C" outputId="dda42a88-8138-4d6d-d683-6b390df89be1">

``` python
print('R^2 Score: ', r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 Score:  -47.6739351631383

</div>

</div>

<div class="cell code" execution_count="342"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="iZdC2s_OYK93" outputId="1227e37a-46fe-4ff1-eba7-5f5269622a73">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  5205.3834886693485

</div>

</div>

<div class="cell code" execution_count="343" id="Jrjq7H1aiCYs">

``` python
matrix.append(["Полиномна регресија со степен 4",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="uDrTdQpHYcG6">

*Овој модел не дава добри резултати*

</div>

<div class="cell markdown" id="Uy9DMChH_u_J">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="c03Oc5EO_wrN">

### 8.2.Полиномна регресија со степен 2

</div>

<div class="cell code" execution_count="344" id="ImVpdBZTY-T_">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="345" id="JCURY4LAYyJd">

``` python
poly_reg=PolynomialFeatures(2)
X_train_poly=poly_reg.fit_transform(X_train)
poly_reg.fit(X_train_poly,Y_train)

lin_reg=LinearRegression()
lin_reg.fit(X_train_poly,Y_train)

X_test_poly=poly_reg.fit_transform(X_test)
y_pred=lin_reg.predict(X_test_poly)

training_score=mean_squared_error(Y_train, lin_reg.predict(X_train_poly))
test_score=mean_squared_error(Y_test,y_pred)
```

</div>

<div class="cell code" execution_count="346"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="x78BtMz7oWtj" outputId="e4301830-ffc3-48f8-d06d-169f459293c4">

``` python
print('R^2 Score: ', r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 Score:  0.9670140517921552

</div>

</div>

<div class="cell code" execution_count="347"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="OjCX-SaZY1fN" outputId="11f0bafc-5edf-4418-f0f1-9d36df922f9b">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  3.5276480026470707

</div>

</div>

<div class="cell code" execution_count="348" id="gB2MqCTziHXk">

``` python
matrix.append(["Полиномна регресија со степен 2",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="w4weo2yXY3Id">

*Овој модел дава добри резултати*

</div>

<div class="cell markdown" id="T1iu4dL2Y_-n">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="oJ9kTUwiaAiW">

## 9.Модел со XGBoost

</div>

<div class="cell code" execution_count="349" id="ng8NZrsbYMON">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="350"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="jyzBGMJnaFMt" outputId="2d717c3c-f94e-4c4a-f01a-c6a2f2698bb5">

``` python
model=XGBRegressor(objective ='reg:linear',
	colsample_bytree = 0.2,
	learning_rate = 0.5,
	max_depth = 4,
	alpha = 5,
	n_estimators = 5)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

<div class="output stream stdout">

    [17:07:00] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

</div>

</div>

<div class="cell code" execution_count="351"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="1r2YskepoYzg" outputId="438aab80-28c7-4933-97e4-94af43725f02">

``` python
print('R^2 Score: ', r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 Score:  0.9531358366786692

</div>

</div>

<div class="cell code" execution_count="352"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="flqKcpPYaK3v" outputId="02e12166-335a-4c1f-d30d-5efe98a3f945">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  5.011839316988365

</div>

</div>

<div class="cell code" execution_count="353" id="Wq7aiXqUiJB5">

``` python
matrix.append(["XGBoost",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="PZ87si24ahfD">

*Овој модел дава солидни резултати*

</div>

<div class="cell markdown" id="yeXEFxv_alkm">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="IS-xODSwaolH">

## 10.Модел со LightGBM

</div>

<div class="cell code" execution_count="354" id="fhkNuU4aaQKu">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="355" id="O-TcKXrEasM1">

``` python
from lightgbm import LGBMRegressor
model=LGBMRegressor(n_estimators = 100)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

</div>

<div class="cell code" execution_count="356"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Iai4qN6hoZlg" outputId="d61d837b-a1d2-4a1e-9eb4-c52d26e54aaa">

``` python
print('R^2 Score: ', r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 Score:  0.9481200048603692

</div>

</div>

<div class="cell code" execution_count="357"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="vDr86Mi-atKQ" outputId="aa02873e-9b06-4b75-a691-2c2103c98515">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  5.548252246031639

</div>

</div>

<div class="cell code" execution_count="358" id="c0jQ843KiLCz">

``` python
matrix.append(["LightGBM",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="fJJJ4kpma6uJ">

*Овој модел дава солидни резултати*

</div>

<div class="cell markdown" id="pT7C6i6ga94Z">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="G1KqwxlmbAmk">

## 11.Модел со CatBoost

</div>

<div class="cell code" execution_count="359" id="QcD9ljm9awJH">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="360"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="8GWSKg5WbIKR" outputId="a8d1f448-6948-4322-9583-e31fa57d6dd6">

``` python
model = CatBoostRegressor(iterations=500,
	learning_rate=0.05,
        depth=10,
        eval_metric='RMSE',
        random_seed = 42,
        bagging_temperature = 0.2,
        od_type='Iter',
        metric_period = 50,
        od_wait=20)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

<div class="output stream stdout">

    0:	learn: 11.3735896	total: 13.7ms	remaining: 6.83s
    50:	learn: 2.4151251	total: 375ms	remaining: 3.3s
    100:	learn: 1.3066763	total: 732ms	remaining: 2.89s
    150:	learn: 0.9758470	total: 1.06s	remaining: 2.45s
    200:	learn: 0.8153665	total: 1.39s	remaining: 2.07s
    250:	learn: 0.6811388	total: 1.72s	remaining: 1.71s
    300:	learn: 0.5894983	total: 2.04s	remaining: 1.35s
    350:	learn: 0.5103431	total: 2.37s	remaining: 1s
    400:	learn: 0.4406168	total: 2.67s	remaining: 658ms
    450:	learn: 0.3890183	total: 2.97s	remaining: 323ms
    499:	learn: 0.3417550	total: 3.28s	remaining: 0us

</div>

</div>

<div class="cell code" execution_count="361"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="FZdadZ7woaf8" outputId="8c199f2b-cfe6-4e35-8c86-d8b801b6438d">

``` python
print('R^2 Score: ', r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 Score:  0.9481200048603692

</div>

</div>

<div class="cell code" execution_count="362"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="arxm5rmcbUAX" outputId="71f7620e-068b-4b3e-f43a-fe68fbb38e9c">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  5.548252246031639

</div>

</div>

<div class="cell code" execution_count="363" id="uj3kDmJfiODk">

``` python
matrix.append(["CatBoost",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="pD1Mi6z4bpXB">

*Овој модел дава слично добри резултати како и претходните.*

</div>

<div class="cell markdown" id="SnVghyyLbtiJ">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="-5bMA98h76x4">

## 12.Модели со невронски мрежи

</div>

<div class="cell markdown" id="8Mx-lUU9Cvlt">

### 12.1.Прв модел

</div>

<div class="cell code" execution_count="364" id="imcs04srAqR-">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="365"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="C8evRRPMbcv-" outputId="65e43dbc-971e-4cb8-d0fb-c51f7e9c62dc">

``` python
model = Sequential()

model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='linear'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=10, batch_size=10,  verbose=1, validation_split=0.2)
y_pred = model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

<div class="output stream stdout">

    Epoch 1/10
    14/14 [==============================] - 1s 16ms/step - loss: 330.1874 - val_loss: 148.9009
    Epoch 2/10
    14/14 [==============================] - 0s 5ms/step - loss: 116.9269 - val_loss: 49.2527
    Epoch 3/10
    14/14 [==============================] - 0s 5ms/step - loss: 78.0457 - val_loss: 50.3760
    Epoch 4/10
    14/14 [==============================] - 0s 7ms/step - loss: 61.1420 - val_loss: 27.7927
    Epoch 5/10
    14/14 [==============================] - 0s 7ms/step - loss: 35.1721 - val_loss: 15.0901
    Epoch 6/10
    14/14 [==============================] - 0s 6ms/step - loss: 18.7570 - val_loss: 6.5911
    Epoch 7/10
    14/14 [==============================] - 0s 6ms/step - loss: 9.4966 - val_loss: 3.2239
    Epoch 8/10
    14/14 [==============================] - 0s 6ms/step - loss: 5.3260 - val_loss: 3.0326
    Epoch 9/10
    14/14 [==============================] - 0s 7ms/step - loss: 4.0234 - val_loss: 3.4047
    Epoch 10/10
    14/14 [==============================] - 0s 6ms/step - loss: 3.4479 - val_loss: 4.4895

</div>

</div>

<div class="cell code" execution_count="366" id="smfzymOYAi5x">

``` python
y_pred = model.predict(X_test)
```

</div>

<div class="cell code" execution_count="367"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:281}"
id="V31P0975As7P" outputId="2c236f00-14c9-4f76-8522-f4a694532986">

``` python
plt.plot(Y_test, color = 'green', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.grid(alpha = 0.3)
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```

<div class="output display_data">

![](040ad4f445e319305cd42cfc0acce6ebc87b00f0.png)

</div>

</div>

<div class="cell code" execution_count="368"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="7iBn1njFAz4-" outputId="a5a3d6cb-0898-462e-8d13-067c86bd4a47">

``` python
print("R^2 score: ", r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 score:  0.9858672682163825

</div>

</div>

<div class="cell code" execution_count="369"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="bm-i4XZ_BDlj" outputId="260b33ad-e804-476a-8b5c-37abce4b6503">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  1.5114103355248778

</div>

</div>

<div class="cell code" execution_count="370" id="sbsJInP_iRl7">

``` python
matrix.append(["Невронска мрежа #1",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="y0BiGxmhDgG3">

*Овој модел дава солидни резултати*

</div>

<div class="cell markdown" id="EJQUJSADCx4U">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="ejD50XAHCz_R">

### 12.2.Втор модел

</div>

<div class="cell code" execution_count="371" id="MT1_rCCgBW_Q">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="372"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="sO5RSBFEC45H" outputId="5858668f-829e-4801-c2e3-6506287722d9">

``` python
model = Sequential()

model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2, input_shape=(256,)))
model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=30, batch_size=20,  verbose=1, validation_split=0.2)
y_pred = model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

<div class="output stream stdout">

    Epoch 1/30
    7/7 [==============================] - 1s 26ms/step - loss: 444.0495 - val_loss: 408.4590
    Epoch 2/30
    7/7 [==============================] - 0s 6ms/step - loss: 377.5623 - val_loss: 305.3311
    Epoch 3/30
    7/7 [==============================] - 0s 6ms/step - loss: 245.6134 - val_loss: 158.7994
    Epoch 4/30
    7/7 [==============================] - 0s 7ms/step - loss: 119.9106 - val_loss: 57.6062
    Epoch 5/30
    7/7 [==============================] - 0s 6ms/step - loss: 88.4978 - val_loss: 49.9743
    Epoch 6/30
    7/7 [==============================] - 0s 6ms/step - loss: 78.5773 - val_loss: 35.5146
    Epoch 7/30
    7/7 [==============================] - 0s 6ms/step - loss: 59.8658 - val_loss: 35.7939
    Epoch 8/30
    7/7 [==============================] - 0s 6ms/step - loss: 53.5467 - val_loss: 28.6432
    Epoch 9/30
    7/7 [==============================] - 0s 9ms/step - loss: 42.4716 - val_loss: 20.9596
    Epoch 10/30
    7/7 [==============================] - 0s 7ms/step - loss: 33.1474 - val_loss: 15.3638
    Epoch 11/30
    7/7 [==============================] - 0s 7ms/step - loss: 24.8783 - val_loss: 12.2049
    Epoch 12/30
    7/7 [==============================] - 0s 7ms/step - loss: 17.7262 - val_loss: 7.0362
    Epoch 13/30
    7/7 [==============================] - 0s 7ms/step - loss: 12.5915 - val_loss: 4.7648
    Epoch 14/30
    7/7 [==============================] - 0s 7ms/step - loss: 9.7026 - val_loss: 4.6003
    Epoch 15/30
    7/7 [==============================] - 0s 7ms/step - loss: 6.4725 - val_loss: 3.0873
    Epoch 16/30
    7/7 [==============================] - 0s 10ms/step - loss: 5.9879 - val_loss: 3.1112
    Epoch 17/30
    7/7 [==============================] - 0s 7ms/step - loss: 4.7659 - val_loss: 3.9210
    Epoch 18/30
    7/7 [==============================] - 0s 7ms/step - loss: 5.6291 - val_loss: 3.6142
    Epoch 19/30
    7/7 [==============================] - 0s 7ms/step - loss: 3.9882 - val_loss: 3.2134
    Epoch 20/30
    7/7 [==============================] - 0s 8ms/step - loss: 4.5868 - val_loss: 3.1144
    Epoch 21/30
    7/7 [==============================] - 0s 11ms/step - loss: 4.0767 - val_loss: 3.2395
    Epoch 22/30
    7/7 [==============================] - 0s 7ms/step - loss: 4.4404 - val_loss: 3.3827
    Epoch 23/30
    7/7 [==============================] - 0s 7ms/step - loss: 3.3705 - val_loss: 3.3818
    Epoch 24/30
    7/7 [==============================] - 0s 8ms/step - loss: 3.4630 - val_loss: 3.7351
    Epoch 25/30
    7/7 [==============================] - 0s 7ms/step - loss: 3.2340 - val_loss: 3.6890
    Epoch 26/30
    7/7 [==============================] - 0s 8ms/step - loss: 3.4434 - val_loss: 3.5617
    Epoch 27/30
    7/7 [==============================] - 0s 9ms/step - loss: 2.9105 - val_loss: 3.9884
    Epoch 28/30
    7/7 [==============================] - 0s 7ms/step - loss: 3.1350 - val_loss: 3.4603
    Epoch 29/30
    7/7 [==============================] - 0s 8ms/step - loss: 3.0801 - val_loss: 3.4248
    Epoch 30/30
    7/7 [==============================] - 0s 7ms/step - loss: 3.0430 - val_loss: 3.4323

</div>

</div>

<div class="cell code" execution_count="373" id="xj7j-rgJDBbJ">

``` python
y_pred = model.predict(X_test)
```

</div>

<div class="cell code" execution_count="374"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:281}"
id="lIPfJvwKDCvL" outputId="553f821a-b052-4c51-a209-7a81c4cabcf7">

``` python
plt.plot(Y_test, color = 'green', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.grid(alpha = 0.3)
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```

<div class="output display_data">

![](ea1bad890b441b097dfa3b13fd4b9fc545dab54f.png)

</div>

</div>

<div class="cell code" execution_count="375"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="2X41ANi8DDVi" outputId="0b06661b-451e-47c6-d61d-bb0435efa79f">

``` python
print("R^2 score: ", r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 score:  0.9859833873914288

</div>

</div>

<div class="cell code" execution_count="376"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="0HnSh61NDEhp" outputId="86279a22-eb3b-4c26-b09d-2e8917bfa46a">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  1.4989920908426309

</div>

</div>

<div class="cell code" execution_count="377" id="XQtkqeZEiWKT">

``` python
matrix.append(["Невронска мрежа #2",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="tBYJ2JkRDkct">

*Овој модел дава добри резултати*

</div>

<div class="cell markdown" id="eh-hDXxxDnFm">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="ZYfiYlwiDqf0">

### 12.3.Трет модел

</div>

<div class="cell code" execution_count="378" id="nlRNsz_RDIwk">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="379"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="ke3wiK3SEFkX" outputId="a8520b4b-b8eb-48a9-c402-41ff6291502b">

``` python
model = Sequential()

model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2, input_shape=(128,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(812, activation='relu'))
model.add(Dropout(0.2, input_shape=(512,)))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=50, batch_size=20,  verbose=1, validation_split=0.2)
y_pred = model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

<div class="output stream stdout">

    Epoch 1/50
    7/7 [==============================] - 1s 29ms/step - loss: 440.1865 - val_loss: 393.7437
    Epoch 2/50
    7/7 [==============================] - 0s 8ms/step - loss: 332.2568 - val_loss: 239.5009
    Epoch 3/50
    7/7 [==============================] - 0s 10ms/step - loss: 168.5382 - val_loss: 79.5765
    Epoch 4/50
    7/7 [==============================] - 0s 10ms/step - loss: 113.8502 - val_loss: 50.6854
    Epoch 5/50
    7/7 [==============================] - 0s 8ms/step - loss: 79.4390 - val_loss: 55.0958
    Epoch 6/50
    7/7 [==============================] - 0s 8ms/step - loss: 71.7142 - val_loss: 42.1065
    Epoch 7/50
    7/7 [==============================] - 0s 8ms/step - loss: 52.2623 - val_loss: 29.8301
    Epoch 8/50
    7/7 [==============================] - 0s 8ms/step - loss: 52.4247 - val_loss: 22.7438
    Epoch 9/50
    7/7 [==============================] - 0s 8ms/step - loss: 37.2855 - val_loss: 22.2504
    Epoch 10/50
    7/7 [==============================] - 0s 9ms/step - loss: 23.6565 - val_loss: 10.6570
    Epoch 11/50
    7/7 [==============================] - 0s 8ms/step - loss: 21.8197 - val_loss: 8.2666
    Epoch 12/50
    7/7 [==============================] - 0s 9ms/step - loss: 11.7379 - val_loss: 3.3939
    Epoch 13/50
    7/7 [==============================] - 0s 8ms/step - loss: 8.2003 - val_loss: 2.0813
    Epoch 14/50
    7/7 [==============================] - 0s 9ms/step - loss: 7.3704 - val_loss: 3.1338
    Epoch 15/50
    7/7 [==============================] - 0s 8ms/step - loss: 6.4213 - val_loss: 2.4228
    Epoch 16/50
    7/7 [==============================] - 0s 10ms/step - loss: 6.8712 - val_loss: 3.7186
    Epoch 17/50
    7/7 [==============================] - 0s 8ms/step - loss: 5.9583 - val_loss: 2.2602
    Epoch 18/50
    7/7 [==============================] - 0s 8ms/step - loss: 5.7014 - val_loss: 2.3337
    Epoch 19/50
    7/7 [==============================] - 0s 10ms/step - loss: 4.0600 - val_loss: 2.5393
    Epoch 20/50
    7/7 [==============================] - 0s 8ms/step - loss: 5.1509 - val_loss: 2.9147
    Epoch 21/50
    7/7 [==============================] - 0s 9ms/step - loss: 5.7713 - val_loss: 5.3793
    Epoch 22/50
    7/7 [==============================] - 0s 9ms/step - loss: 6.2975 - val_loss: 3.5930
    Epoch 23/50
    7/7 [==============================] - 0s 9ms/step - loss: 4.5844 - val_loss: 3.4714
    Epoch 24/50
    7/7 [==============================] - 0s 10ms/step - loss: 3.0380 - val_loss: 3.3390
    Epoch 25/50
    7/7 [==============================] - 0s 8ms/step - loss: 4.6254 - val_loss: 3.3557
    Epoch 26/50
    7/7 [==============================] - 0s 9ms/step - loss: 3.7222 - val_loss: 3.8413
    Epoch 27/50
    7/7 [==============================] - 0s 8ms/step - loss: 4.1812 - val_loss: 3.8396
    Epoch 28/50
    7/7 [==============================] - 0s 12ms/step - loss: 4.7316 - val_loss: 3.8394
    Epoch 29/50
    7/7 [==============================] - 0s 8ms/step - loss: 4.0894 - val_loss: 3.7310
    Epoch 30/50
    7/7 [==============================] - 0s 9ms/step - loss: 4.0904 - val_loss: 3.5209
    Epoch 31/50
    7/7 [==============================] - 0s 8ms/step - loss: 2.8321 - val_loss: 3.6804
    Epoch 32/50
    7/7 [==============================] - 0s 8ms/step - loss: 3.6589 - val_loss: 3.2205
    Epoch 33/50
    7/7 [==============================] - 0s 10ms/step - loss: 3.3787 - val_loss: 3.0400
    Epoch 34/50
    7/7 [==============================] - 0s 13ms/step - loss: 3.8068 - val_loss: 3.7936
    Epoch 35/50
    7/7 [==============================] - 0s 7ms/step - loss: 4.5065 - val_loss: 3.0700
    Epoch 36/50
    7/7 [==============================] - 0s 7ms/step - loss: 3.6468 - val_loss: 4.6607
    Epoch 37/50
    7/7 [==============================] - 0s 8ms/step - loss: 4.2712 - val_loss: 5.3724
    Epoch 38/50
    7/7 [==============================] - 0s 7ms/step - loss: 5.1000 - val_loss: 6.0282
    Epoch 39/50
    7/7 [==============================] - 0s 8ms/step - loss: 4.2118 - val_loss: 3.3688
    Epoch 40/50
    7/7 [==============================] - 0s 7ms/step - loss: 4.5308 - val_loss: 3.1255
    Epoch 41/50
    7/7 [==============================] - 0s 9ms/step - loss: 3.6541 - val_loss: 3.1383
    Epoch 42/50
    7/7 [==============================] - 0s 8ms/step - loss: 3.3937 - val_loss: 3.3016
    Epoch 43/50
    7/7 [==============================] - 0s 8ms/step - loss: 3.9458 - val_loss: 2.8990
    Epoch 44/50
    7/7 [==============================] - 0s 7ms/step - loss: 4.5707 - val_loss: 3.6017
    Epoch 45/50
    7/7 [==============================] - 0s 8ms/step - loss: 5.2518 - val_loss: 3.6674
    Epoch 46/50
    7/7 [==============================] - 0s 8ms/step - loss: 3.3125 - val_loss: 2.9959
    Epoch 47/50
    7/7 [==============================] - 0s 8ms/step - loss: 4.8239 - val_loss: 2.9770
    Epoch 48/50
    7/7 [==============================] - 0s 9ms/step - loss: 4.5752 - val_loss: 3.7437
    Epoch 49/50
    7/7 [==============================] - 0s 8ms/step - loss: 4.4203 - val_loss: 2.9593
    Epoch 50/50
    7/7 [==============================] - 0s 9ms/step - loss: 3.3688 - val_loss: 3.2939

</div>

</div>

<div class="cell code" execution_count="380" id="MrvfITP3EIXK">

``` python
y_pred = model.predict(X_test)
```

</div>

<div class="cell code" execution_count="381"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:281}"
id="wdvUduroEKQS" outputId="39478739-72d2-40da-9518-36e80665f8cf">

``` python
plt.plot(Y_test, color = 'green', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.grid(alpha = 0.3)
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```

<div class="output display_data">

![](8ed463729ce1320ed2be622f749152e9b4ba7775.png)

</div>

</div>

<div class="cell code" execution_count="382"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="reqxADOAELlG" outputId="e25b0166-f725-40f3-c700-5c20b1c12b3b">

``` python
print("R^2 score: ", r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 score:  0.9820403601589779

</div>

</div>

<div class="cell code" execution_count="383"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="azHWZia-EM3J" outputId="bededf80-c278-4eee-c97c-ff35c843c925">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  1.9206750466665412

</div>

</div>

<div class="cell code" execution_count="384" id="zLsdplMoiXsw">

``` python
matrix.append(["Невронска мрежа #3",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="F5G0oeYvE1XQ">

*Овој модел дава подобри резултати од претходните два модели со
невронски мрежи*

</div>

<div class="cell markdown" id="JUnlrqeME97C">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="J5LRmiuLFDvj">

### 12.4.Четврт модел

</div>

<div class="cell code" execution_count="385" id="PF5JC19kEtrZ">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="386"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="om5ErCRaFJJT" outputId="c2a295f4-dd8f-49c1-9b52-eeb3dac4c9b3">

``` python
model = Sequential()

model.add(Dense(64, input_dim=4, activation='linear'))
model.add(Dropout(0.2, input_shape=(128,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2, input_shape=(256,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2, input_shape=(512,)))
model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=100, batch_size=50,  verbose=1, validation_split=0.2)
y_pred = model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

<div class="output stream stdout">

    Epoch 1/100
    3/3 [==============================] - 1s 96ms/step - loss: 420.8204 - val_loss: 324.8066
    Epoch 2/100
    3/3 [==============================] - 0s 27ms/step - loss: 264.2183 - val_loss: 147.7545
    Epoch 3/100
    3/3 [==============================] - 0s 21ms/step - loss: 137.9233 - val_loss: 184.0346
    Epoch 4/100
    3/3 [==============================] - 0s 20ms/step - loss: 174.5963 - val_loss: 110.8116
    Epoch 5/100
    3/3 [==============================] - 0s 19ms/step - loss: 113.9720 - val_loss: 101.8778
    Epoch 6/100
    3/3 [==============================] - 0s 19ms/step - loss: 94.7111 - val_loss: 88.4798
    Epoch 7/100
    3/3 [==============================] - 0s 22ms/step - loss: 76.9885 - val_loss: 55.7771
    Epoch 8/100
    3/3 [==============================] - 0s 23ms/step - loss: 57.1538 - val_loss: 44.1228
    Epoch 9/100
    3/3 [==============================] - 0s 19ms/step - loss: 47.8985 - val_loss: 25.9867
    Epoch 10/100
    3/3 [==============================] - 0s 20ms/step - loss: 28.8975 - val_loss: 15.5729
    Epoch 11/100
    3/3 [==============================] - 0s 22ms/step - loss: 22.9940 - val_loss: 10.1308
    Epoch 12/100
    3/3 [==============================] - 0s 19ms/step - loss: 18.5925 - val_loss: 11.7994
    Epoch 13/100
    3/3 [==============================] - 0s 26ms/step - loss: 21.1534 - val_loss: 14.5971
    Epoch 14/100
    3/3 [==============================] - 0s 22ms/step - loss: 16.6676 - val_loss: 13.0103
    Epoch 15/100
    3/3 [==============================] - 0s 22ms/step - loss: 19.0054 - val_loss: 11.1421
    Epoch 16/100
    3/3 [==============================] - 0s 20ms/step - loss: 20.4043 - val_loss: 8.8484
    Epoch 17/100
    3/3 [==============================] - 0s 20ms/step - loss: 15.1089 - val_loss: 9.2479
    Epoch 18/100
    3/3 [==============================] - 0s 21ms/step - loss: 17.2931 - val_loss: 6.6254
    Epoch 19/100
    3/3 [==============================] - 0s 23ms/step - loss: 12.1348 - val_loss: 7.0734
    Epoch 20/100
    3/3 [==============================] - 0s 20ms/step - loss: 12.1968 - val_loss: 5.5257
    Epoch 21/100
    3/3 [==============================] - 0s 21ms/step - loss: 12.4860 - val_loss: 5.5129
    Epoch 22/100
    3/3 [==============================] - 0s 21ms/step - loss: 10.3159 - val_loss: 5.3071
    Epoch 23/100
    3/3 [==============================] - 0s 21ms/step - loss: 12.0136 - val_loss: 4.8743
    Epoch 24/100
    3/3 [==============================] - 0s 23ms/step - loss: 10.8299 - val_loss: 4.5037
    Epoch 25/100
    3/3 [==============================] - 0s 21ms/step - loss: 8.1931 - val_loss: 4.7553
    Epoch 26/100
    3/3 [==============================] - 0s 22ms/step - loss: 5.5237 - val_loss: 3.6467
    Epoch 27/100
    3/3 [==============================] - 0s 20ms/step - loss: 8.6126 - val_loss: 3.7922
    Epoch 28/100
    3/3 [==============================] - 0s 21ms/step - loss: 8.9877 - val_loss: 3.1153
    Epoch 29/100
    3/3 [==============================] - 0s 29ms/step - loss: 8.7638 - val_loss: 3.0796
    Epoch 30/100
    3/3 [==============================] - 0s 19ms/step - loss: 7.4829 - val_loss: 3.1056
    Epoch 31/100
    3/3 [==============================] - 0s 20ms/step - loss: 6.7011 - val_loss: 2.9602
    Epoch 32/100
    3/3 [==============================] - 0s 21ms/step - loss: 7.3349 - val_loss: 2.9976
    Epoch 33/100
    3/3 [==============================] - 0s 24ms/step - loss: 6.0383 - val_loss: 3.1352
    Epoch 34/100
    3/3 [==============================] - 0s 19ms/step - loss: 5.6119 - val_loss: 3.0346
    Epoch 35/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.8267 - val_loss: 3.1169
    Epoch 36/100
    3/3 [==============================] - 0s 20ms/step - loss: 6.3884 - val_loss: 2.7789
    Epoch 37/100
    3/3 [==============================] - 0s 22ms/step - loss: 7.1024 - val_loss: 3.0462
    Epoch 38/100
    3/3 [==============================] - 0s 19ms/step - loss: 6.8425 - val_loss: 4.0966
    Epoch 39/100
    3/3 [==============================] - 0s 22ms/step - loss: 6.4307 - val_loss: 3.1013
    Epoch 40/100
    3/3 [==============================] - 0s 21ms/step - loss: 6.4498 - val_loss: 3.3343
    Epoch 41/100
    3/3 [==============================] - 0s 21ms/step - loss: 4.9419 - val_loss: 4.8782
    Epoch 42/100
    3/3 [==============================] - 0s 21ms/step - loss: 6.3614 - val_loss: 4.5430
    Epoch 43/100
    3/3 [==============================] - 0s 24ms/step - loss: 8.5049 - val_loss: 2.7236
    Epoch 44/100
    3/3 [==============================] - 0s 21ms/step - loss: 8.1143 - val_loss: 3.8314
    Epoch 45/100
    3/3 [==============================] - 0s 20ms/step - loss: 6.0973 - val_loss: 4.8568
    Epoch 46/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.9856 - val_loss: 2.7191
    Epoch 47/100
    3/3 [==============================] - 0s 21ms/step - loss: 5.3708 - val_loss: 4.6476
    Epoch 48/100
    3/3 [==============================] - 0s 21ms/step - loss: 5.6067 - val_loss: 3.5610
    Epoch 49/100
    3/3 [==============================] - 0s 23ms/step - loss: 6.6643 - val_loss: 2.9926
    Epoch 50/100
    3/3 [==============================] - 0s 20ms/step - loss: 4.5201 - val_loss: 3.7920
    Epoch 51/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.7515 - val_loss: 2.9798
    Epoch 52/100
    3/3 [==============================] - 0s 19ms/step - loss: 4.8214 - val_loss: 3.7380
    Epoch 53/100
    3/3 [==============================] - 0s 21ms/step - loss: 5.9489 - val_loss: 3.5798
    Epoch 54/100
    3/3 [==============================] - 0s 21ms/step - loss: 6.0475 - val_loss: 2.8490
    Epoch 55/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.6611 - val_loss: 4.1694
    Epoch 56/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.9288 - val_loss: 4.5472
    Epoch 57/100
    3/3 [==============================] - 0s 19ms/step - loss: 7.4505 - val_loss: 3.2054
    Epoch 58/100
    3/3 [==============================] - 0s 26ms/step - loss: 5.2479 - val_loss: 3.4496
    Epoch 59/100
    3/3 [==============================] - 0s 21ms/step - loss: 3.6302 - val_loss: 4.3813
    Epoch 60/100
    3/3 [==============================] - 0s 25ms/step - loss: 4.9028 - val_loss: 2.7423
    Epoch 61/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.1412 - val_loss: 3.0425
    Epoch 62/100
    3/3 [==============================] - 0s 22ms/step - loss: 4.1683 - val_loss: 2.8327
    Epoch 63/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.7594 - val_loss: 3.3280
    Epoch 64/100
    3/3 [==============================] - 0s 19ms/step - loss: 4.9751 - val_loss: 3.1736
    Epoch 65/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.6147 - val_loss: 2.7328
    Epoch 66/100
    3/3 [==============================] - 0s 21ms/step - loss: 5.1470 - val_loss: 3.1186
    Epoch 67/100
    3/3 [==============================] - 0s 21ms/step - loss: 4.4309 - val_loss: 4.4350
    Epoch 68/100
    3/3 [==============================] - 0s 21ms/step - loss: 4.7826 - val_loss: 3.0421
    Epoch 69/100
    3/3 [==============================] - 0s 27ms/step - loss: 3.9310 - val_loss: 3.3630
    Epoch 70/100
    3/3 [==============================] - 0s 23ms/step - loss: 4.3630 - val_loss: 3.3195
    Epoch 71/100
    3/3 [==============================] - 0s 21ms/step - loss: 5.8244 - val_loss: 2.9849
    Epoch 72/100
    3/3 [==============================] - 0s 20ms/step - loss: 3.5014 - val_loss: 2.9794
    Epoch 73/100
    3/3 [==============================] - 0s 23ms/step - loss: 4.3873 - val_loss: 3.1851
    Epoch 74/100
    3/3 [==============================] - 0s 21ms/step - loss: 4.5613 - val_loss: 2.9058
    Epoch 75/100
    3/3 [==============================] - 0s 24ms/step - loss: 5.8500 - val_loss: 2.9563
    Epoch 76/100
    3/3 [==============================] - 0s 20ms/step - loss: 6.2621 - val_loss: 4.0081
    Epoch 77/100
    3/3 [==============================] - 0s 24ms/step - loss: 5.7893 - val_loss: 5.0961
    Epoch 78/100
    3/3 [==============================] - 0s 21ms/step - loss: 7.5628 - val_loss: 2.6626
    Epoch 79/100
    3/3 [==============================] - 0s 21ms/step - loss: 4.7511 - val_loss: 2.8406
    Epoch 80/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.4968 - val_loss: 3.2991
    Epoch 81/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.2273 - val_loss: 2.6193
    Epoch 82/100
    3/3 [==============================] - 0s 22ms/step - loss: 6.2173 - val_loss: 3.0679
    Epoch 83/100
    3/3 [==============================] - 0s 19ms/step - loss: 6.8227 - val_loss: 2.7238
    Epoch 84/100
    3/3 [==============================] - 0s 19ms/step - loss: 4.5558 - val_loss: 4.2904
    Epoch 85/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.1251 - val_loss: 3.0274
    Epoch 86/100
    3/3 [==============================] - 0s 19ms/step - loss: 4.8990 - val_loss: 2.7060
    Epoch 87/100
    3/3 [==============================] - 0s 20ms/step - loss: 4.6251 - val_loss: 3.2111
    Epoch 88/100
    3/3 [==============================] - 0s 22ms/step - loss: 5.4541 - val_loss: 5.6851
    Epoch 89/100
    3/3 [==============================] - 0s 19ms/step - loss: 4.8133 - val_loss: 2.7720
    Epoch 90/100
    3/3 [==============================] - 0s 19ms/step - loss: 6.1750 - val_loss: 3.2576
    Epoch 91/100
    3/3 [==============================] - 0s 20ms/step - loss: 4.1920 - val_loss: 3.0729
    Epoch 92/100
    3/3 [==============================] - 0s 20ms/step - loss: 4.0724 - val_loss: 2.8716
    Epoch 93/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.4820 - val_loss: 3.2390
    Epoch 94/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.3854 - val_loss: 3.0595
    Epoch 95/100
    3/3 [==============================] - 0s 21ms/step - loss: 4.9948 - val_loss: 2.9503
    Epoch 96/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.8310 - val_loss: 3.0205
    Epoch 97/100
    3/3 [==============================] - 0s 19ms/step - loss: 5.3720 - val_loss: 3.1092
    Epoch 98/100
    3/3 [==============================] - 0s 20ms/step - loss: 5.3444 - val_loss: 3.0019
    Epoch 99/100
    3/3 [==============================] - 0s 20ms/step - loss: 6.2357 - val_loss: 3.4292
    Epoch 100/100
    3/3 [==============================] - 0s 18ms/step - loss: 4.6292 - val_loss: 3.7543

</div>

</div>

<div class="cell code" execution_count="387" id="avnzCP0fFJiM">

``` python
y_pred = model.predict(X_test)
```

</div>

<div class="cell code" execution_count="388"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:281}"
id="739ENHVDFLMD" outputId="3da98b0e-b9fc-49b5-89e5-d71bae1e4eae">

``` python
plt.plot(Y_test, color = 'green', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.grid(alpha = 0.3)
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```

<div class="output display_data">

![](b8a2815d87b4f0bb8406808ca10f781ac994e8e6.png)

</div>

</div>

<div class="cell code" execution_count="389"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="TjnG6hk9FMrf" outputId="6a90a4c1-0014-4615-9d69-266e778563b7">

``` python
print("R^2 score: ", r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 score:  0.9776790528476612

</div>

</div>

<div class="cell code" execution_count="390"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="8UT4g47SFN3z" outputId="51da0bde-64fa-4e44-af87-d55736c58a12">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  2.3870905314891755

</div>

</div>

<div class="cell code" execution_count="391" id="TxYmgZdyiZ1d">

``` python
matrix.append(["Невронска мрежа #4",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="GVPYK1e0WEk_">

*Овој модел дава слични резултати како претходниот*

</div>

<div class="cell markdown" id="yNf8VeYRWHrz">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="Z8J22mN5WR6t">

### 12.5.Петти модел

</div>

<div class="cell code" execution_count="392" id="pBtfJfSZUy-w">

``` python
X_train, X_test, Y_train, Y_test=make_copies()
```

</div>

<div class="cell code" execution_count="393"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="gqR00HMMWWA7" outputId="e482e034-08f4-4bca-8b64-931ea0e14b24">

``` python
model = Sequential()

model.add(Dense(128, input_dim=4, activation='linear'))
model.add(Dropout(0.2, input_shape=(128,)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2, input_shape=(128,)))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2, input_shape=(256,)))
model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=50, batch_size=25,  verbose=1, validation_split=0.2)
y_pred = model.predict(X_test)
test_score=mean_squared_error(Y_test,y_pred)
```

<div class="output stream stdout">

    Epoch 1/50
    6/6 [==============================] - 1s 35ms/step - loss: 389.2433 - val_loss: 286.4176
    Epoch 2/50
    6/6 [==============================] - 0s 10ms/step - loss: 216.6700 - val_loss: 134.1519
    Epoch 3/50
    6/6 [==============================] - 0s 9ms/step - loss: 129.4104 - val_loss: 128.7855
    Epoch 4/50
    6/6 [==============================] - 0s 9ms/step - loss: 106.5442 - val_loss: 73.3540
    Epoch 5/50
    6/6 [==============================] - 0s 9ms/step - loss: 74.9570 - val_loss: 50.2683
    Epoch 6/50
    6/6 [==============================] - 0s 9ms/step - loss: 52.1594 - val_loss: 26.9471
    Epoch 7/50
    6/6 [==============================] - 0s 9ms/step - loss: 31.2964 - val_loss: 17.8398
    Epoch 8/50
    6/6 [==============================] - 0s 9ms/step - loss: 28.0380 - val_loss: 17.5270
    Epoch 9/50
    6/6 [==============================] - 0s 9ms/step - loss: 28.7805 - val_loss: 17.7863
    Epoch 10/50
    6/6 [==============================] - 0s 9ms/step - loss: 24.0980 - val_loss: 14.2169
    Epoch 11/50
    6/6 [==============================] - 0s 9ms/step - loss: 19.4782 - val_loss: 11.0839
    Epoch 12/50
    6/6 [==============================] - 0s 12ms/step - loss: 17.9466 - val_loss: 8.7981
    Epoch 13/50
    6/6 [==============================] - 0s 9ms/step - loss: 15.5707 - val_loss: 7.6976
    Epoch 14/50
    6/6 [==============================] - 0s 8ms/step - loss: 15.9455 - val_loss: 7.2692
    Epoch 15/50
    6/6 [==============================] - 0s 8ms/step - loss: 12.8860 - val_loss: 6.5637
    Epoch 16/50
    6/6 [==============================] - 0s 9ms/step - loss: 11.3316 - val_loss: 5.7703
    Epoch 17/50
    6/6 [==============================] - 0s 8ms/step - loss: 11.1118 - val_loss: 6.0663
    Epoch 18/50
    6/6 [==============================] - 0s 9ms/step - loss: 10.5218 - val_loss: 5.2348
    Epoch 19/50
    6/6 [==============================] - 0s 10ms/step - loss: 8.8965 - val_loss: 5.9675
    Epoch 20/50
    6/6 [==============================] - 0s 8ms/step - loss: 8.2385 - val_loss: 5.8475
    Epoch 21/50
    6/6 [==============================] - 0s 8ms/step - loss: 7.1143 - val_loss: 3.7820
    Epoch 22/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.4438 - val_loss: 5.2702
    Epoch 23/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.6207 - val_loss: 4.8924
    Epoch 24/50
    6/6 [==============================] - 0s 9ms/step - loss: 7.0227 - val_loss: 4.7869
    Epoch 25/50
    6/6 [==============================] - 0s 11ms/step - loss: 7.5404 - val_loss: 3.9180
    Epoch 26/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.9163 - val_loss: 4.4529
    Epoch 27/50
    6/6 [==============================] - 0s 9ms/step - loss: 5.8974 - val_loss: 3.5603
    Epoch 28/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.7639 - val_loss: 6.5883
    Epoch 29/50
    6/6 [==============================] - 0s 9ms/step - loss: 8.2217 - val_loss: 3.9370
    Epoch 30/50
    6/6 [==============================] - 0s 11ms/step - loss: 6.7701 - val_loss: 3.2531
    Epoch 31/50
    6/6 [==============================] - 0s 9ms/step - loss: 5.0072 - val_loss: 3.3988
    Epoch 32/50
    6/6 [==============================] - 0s 9ms/step - loss: 3.6655 - val_loss: 4.4376
    Epoch 33/50
    6/6 [==============================] - 0s 10ms/step - loss: 6.4183 - val_loss: 3.4790
    Epoch 34/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.0545 - val_loss: 3.7338
    Epoch 35/50
    6/6 [==============================] - 0s 10ms/step - loss: 5.4339 - val_loss: 3.5432
    Epoch 36/50
    6/6 [==============================] - 0s 11ms/step - loss: 5.1600 - val_loss: 3.7472
    Epoch 37/50
    6/6 [==============================] - 0s 8ms/step - loss: 5.9578 - val_loss: 3.2452
    Epoch 38/50
    6/6 [==============================] - 0s 8ms/step - loss: 6.3001 - val_loss: 4.7462
    Epoch 39/50
    6/6 [==============================] - 0s 9ms/step - loss: 5.3833 - val_loss: 3.1975
    Epoch 40/50
    6/6 [==============================] - 0s 9ms/step - loss: 5.7302 - val_loss: 3.1228
    Epoch 41/50
    6/6 [==============================] - 0s 9ms/step - loss: 4.4206 - val_loss: 2.6367
    Epoch 42/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.2189 - val_loss: 2.9702
    Epoch 43/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.0763 - val_loss: 2.7659
    Epoch 44/50
    6/6 [==============================] - 0s 10ms/step - loss: 6.2162 - val_loss: 2.8729
    Epoch 45/50
    6/6 [==============================] - 0s 9ms/step - loss: 5.5165 - val_loss: 2.6482
    Epoch 46/50
    6/6 [==============================] - 0s 9ms/step - loss: 5.8361 - val_loss: 2.5463
    Epoch 47/50
    6/6 [==============================] - 0s 11ms/step - loss: 4.6456 - val_loss: 4.2698
    Epoch 48/50
    6/6 [==============================] - 0s 9ms/step - loss: 5.8986 - val_loss: 2.9811
    Epoch 49/50
    6/6 [==============================] - 0s 9ms/step - loss: 6.3382 - val_loss: 3.5420
    Epoch 50/50
    6/6 [==============================] - 0s 9ms/step - loss: 7.4437 - val_loss: 3.4393

</div>

</div>

<div class="cell code" execution_count="394" id="NZf6s8zDWjaO">

``` python
y_pred = model.predict(X_test)
```

</div>

<div class="cell code" execution_count="395"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:281}"
id="mN3C4MO0Wk7G" outputId="1a04b4ac-50c5-4e27-cff8-ba52a5f4c6b2">

``` python
plt.plot(Y_test, color = 'green', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.grid(alpha = 0.3)
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```

<div class="output display_data">

![](1a667ea6c4966677db07b9eb724522a6b454057b.png)

</div>

</div>

<div class="cell code" execution_count="396"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="JijhRIrCWm43" outputId="2e5d4a73-dfad-425f-e820-fe84dababd66">

``` python
print("R^2 score: ", r2_score(Y_test,y_pred))
```

<div class="output stream stdout">

    R^2 score:  0.9763901289653097

</div>

</div>

<div class="cell code" execution_count="397"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="hp3x6JSpWoWH" outputId="e8d48884-18c6-496c-d4d0-660e6cb70b62">

``` python
print("Mean squared error: ",test_score)
```

<div class="output stream stdout">

    Mean squared error:  2.5249331586130412

</div>

</div>

<div class="cell code" execution_count="398" id="insz0WZ3ibo3">

``` python
matrix.append(["Невронска мрежа #5",r2_score(Y_test,y_pred).astype(float),mean_squared_error(Y_test,y_pred)])
```

</div>

<div class="cell markdown" id="sGHoHWh4WwqG">

*Овој модел е за нијанса полош од претходните два*

</div>

<div class="cell markdown" id="3XJrK5Z7W16K">

------------------------------------------------------------------------

</div>

<div class="cell markdown" id="ju9tdJl8YDdX">

## 13.Заклучок

</div>

<div class="cell markdown" id="ZnZtvTEhgCC2">

Во рамките на овој проект беше обработено податочното множество за
процентуалната невработеност на млади лица (на возраст од 15 до 24
години) во 220 држави низ светот во периодот од 2010 до 2014 година.

</div>

<div class="cell markdown" id="HCnZgu2IgdF7">

Најпрво беше вчитан и прочистен дата сетот, а потоа беа направени
различни дескриптивни и визуелни анализи над податоците. Потоа датасетот
беше поделен на соодветни множества за тренирање и тестирање, при што
таргет променливата беше процентот на невработени млади лица во 2014
година.

</div>

<div class="cell markdown" id="DUbQ02lbgs08">

Во продолжение беа искористени различни модели од машинско учење за
предвидување на вредностите, меѓу кои и: линеарна регресија, полиномна
регресија, XGBoost, LightGBM, CatBoost и модели со невронски мрежи.

</div>

<div class="cell markdown" id="JrmZ7X8mmJiu">

За евалуација на моделите се користеше R^2 Score и Mean Squared Error.

</div>

<div class="cell markdown" id="6AGNLKWIg8cr">

Во продолжение се сумирани резултатите и перформансите со сите модели:

</div>

<div class="cell code" execution_count="400"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="V4RqGx78hOfo" outputId="609f3764-4191-4144-832a-ad1a6b7973a0">

``` python
result_df=pd.DataFrame(np.array(matrix),columns=['Модел', 'R^2 Score', 'Mean Squared Error'])
print(result_df)
```

<div class="output stream stdout">

                                  Модел           R^2 Score  Mean Squared Error
    0                Линеарна регресија  0.9910553470777943  0.9565766251911539
    1   Полиномна регресија со степен 4   -47.6739351631383  5205.3834886693485
    2   Полиномна регресија со степен 2  0.9670140517921552  3.5276480026470707
    3                           XGBoost  0.9531358366786692   5.011839316988365
    4                          LightGBM  0.9481200048603692   5.548252246031639
    5                          CatBoost  0.9481200048603692   5.548252246031639
    6                Невронска мрежа #1  0.9858672682163825  1.5114103355248778
    7                Невронска мрежа #2  0.9859833873914288  1.4989920908426309
    8                Невронска мрежа #3  0.9820403601589779  1.9206750466665412
    9                Невронска мрежа #4  0.9776790528476612  2.3870905314891755
    10               Невронска мрежа #5  0.9763901289653097  2.5249331586130412

</div>

</div>

<div class="cell markdown" id="4xSHw_rPjlxH">

Од горната табела може да се заклучи дека најдобри резултати даваат
моделите со линеарна регресија и невронските мрежи, додека пак најлош
модел е тој со полиномна регресија од степен 4.

</div>

<div class="cell markdown" id="4VKGqzkopheT">

Најдобар модел е моделот со линеарна регресија кој природно одговара на
проблемот што се разгледува. Уште од визулени анализи можеше да се
забележи дека постои јака линеарна зависност помеѓу променливите што
налага дека линеарната регресија е погодна за решавање на овој проблем.

</div>

<div class="cell markdown" id="_SfH52FwmjlL">

Најдобриот модел од невронските мрежи (невронска мрежа #2) е составен од
секвенцијална невронска мрежба во која има три Dense и еден Dropout слој
во кои има различен број на неврони (128 или 256), со една единствена
активациска функција (relu), проследени со краен Dense слој со големина
1 и линеарна активациска функција.

За компајлирање на моделот е искористен adam оптимизаторот и mean
squared error функцијата на грешка. Направени се 30 епохи и секој batch
има големина 20.

</div>

<div class="cell markdown" id="_qcXyzEBnPT6">

Моделот со линеарна регресија може да се користи за проценка на
процентуалната невработеност на младите лица и во следните години
(2015-денес), притоа најдобро би било на почетокот од годината да се
додадат вистинските вредности од претходната година со цел да се добијат
уште подобри перформанси.

</div>

<div class="cell markdown" id="415ESPIGqepV">

*Забелешка: при различни поделби на податоците се добиваат различни
перформанси на моделите, но моделите со линеарна регресија и невронските
мрежи секогаш даваат убедливо најдобри резултати, додека пак моделот со
полиномна регресија од степен 4 секогаш дава најлоши резултати.*

</div>

<div class="cell markdown" id="4Px4Zt6Nq0hY">

Виктор Мегленовски - 191001

</div>

<div class="cell markdown" id="c0nSirlrq8mO">

ФИНКИ - 05/2022

</div>

<div class="cell markdown" id="bV0hZDAanjG-">

------------------------------------------------------------------------

</div>
