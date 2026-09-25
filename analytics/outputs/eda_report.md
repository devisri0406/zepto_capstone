# EDA Report

## Initial profile

Shape: (891, 15)


### df.info()

<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    str     
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    str     
 8   class        891 non-null    category
 9   who          891 non-null    str     
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    str     
 13  alive        891 non-null    str     
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), str(5)
memory usage: 80.7 KB


### df.describe()

          survived      pclass   sex         age       sibsp       parch        fare embarked  class  who adult_male deck  embark_town alive alone
count   891.000000  891.000000   891  714.000000  891.000000  891.000000  891.000000      889    891  891        891  203          889   891   891
unique         NaN         NaN     2         NaN         NaN         NaN         NaN        3      3    3          2    7            3     2     2
top            NaN         NaN  male         NaN         NaN         NaN         NaN        S  Third  man       True    C  Southampton    no  True
freq           NaN         NaN   577         NaN         NaN         NaN         NaN      644    491  537        537   59          644   549   537
mean      0.383838    2.308642   NaN   29.699118    0.523008    0.381594   32.204208      NaN    NaN  NaN        NaN  NaN          NaN   NaN   NaN
std       0.486592    0.836071   NaN   14.526497    1.102743    0.806057   49.693429      NaN    NaN  NaN        NaN  NaN          NaN   NaN   NaN
min       0.000000    1.000000   NaN    0.420000    0.000000    0.000000    0.000000      NaN    NaN  NaN        NaN  NaN          NaN   NaN   NaN
25%       0.000000    2.000000   NaN   20.125000    0.000000    0.000000    7.910400      NaN    NaN  NaN        NaN  NaN          NaN   NaN   NaN
50%       0.000000    3.000000   NaN   28.000000    0.000000    0.000000   14.454200      NaN    NaN  NaN        NaN  NaN          NaN   NaN   NaN
75%       1.000000    3.000000   NaN   38.000000    1.000000    0.000000   31.000000      NaN    NaN  NaN        NaN  NaN          NaN   NaN   NaN
max       1.000000    3.000000   NaN   80.000000    8.000000    6.000000  512.329200      NaN    NaN  NaN        NaN  NaN          NaN   NaN   NaN

## Missing-value percentages

- `age`: 19.87% — Impute the missing values.

- `embarked`: 0.22% — Drop rows with missing values in this column.

- `deck`: 77.22% — High missingness: drop the column or encode missing as a category.

- `embark_town`: 0.22% — Drop rows with missing values in this column.


## Cleaning decision

Columns below 5% missingness had rows dropped; columns from 5% to 30% were imputed; the high-missingness `deck`/cabin-derived field was excluded from the working EDA dataset because its missingness is too high for reliable imputation and it is not required by the modeling feature set.


## Univariate analysis

- Age IQR outliers: 65

- Fare IQR outliers: 114

- Fare mean: 32.0967; median: 14.4542; mode: 8.0500.

- Fare distribution conclusion from ordering: right-skewed.


## Survival rates


### By sex

sex
female    74.04
male      18.89


### By pclass

pclass
1    62.62
2    47.28
3    24.24


### By sex and pclass

sex     pclass
female  1         96.74
        2         92.11
        3         50.00
male    1         36.89
        2         15.74
        3         13.54

## Two strongest absolute correlations

- `pclass` vs `fare`: correlation = -0.5482

- `sibsp` vs `parch`: correlation = 0.4145


## Multivariate chart interpretations

1. **Survival by sex and class:** The grouped survival-rate chart shows how survival varies jointly by sex and passenger class. The exact pattern should be read from the plotted rates rather than assumed before execution.


2. **Age, survival and sex:** The box plot compares age distributions for survivors and non-survivors within sex groups, helping identify whether age composition differs between outcomes.


3. **Age versus fare:** The scatter plot combines fare, age, survival and class. It provides a multivariate view of how passenger characteristics and ticket class co-occur with survival.


4. **Class and survival:** The point plot shows survival probability by passenger class separately for sex, making the interaction between class and sex visible.



## Standardization sanity check

Before means/stds:
            age       fare
mean  29.315152  32.096681
std   12.984932  49.697504


After means/stds:
             age_z        fare_z
mean  2.717486e-16  1.398706e-16
std   1.000563e+00  1.000563e+00
