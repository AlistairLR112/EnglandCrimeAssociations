
# Association Rule Mining for reported Crimes in England
The aim here is to see if there are any associations between the reported aspects of crime, such as Month of Year, Location, Crime type etc.
This will be done in Pyspark due to the size of the data but it will still be possible to execute on a local cluster.

The data can be downloaded from here: https://data.police.uk/data/.

The date range for this data is September 2015 - August 2018 and all constabularies in England were selected (excluding British Transport Police)

## What is an Association Rule?
*Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness. This rule-based approach also generates new rules as it analyzes more data. The ultimate goal, assuming a large enough dataset, is to help a machine mimic the human brain’s feature extraction and abstract association capabilities from new uncategorized data.*

We will be looking for rules with a high level of confidence

*Confidence is an indication of how often the rule has been found to be true... Confidence can be interpreted as an estimate of the conditional probability*


```python
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import tabulate
import seaborn as sns
```


```python
%load_ext autoreload
%autoreload 2
```

#### Set up Spark
Running Spark locally using 6 out of 8 cores


```python
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
```


```python
spark = SparkSession.builder\
        .master("local[6]")\
        .appName("Crime Assocations")\
        .getOrCreate()
```


```python
sc = spark.sparkContext
# Set up a SQL Context
sqlCtx = SQLContext(sc)
```


```python
#sc.stop()
```

## Load Data into Spark


```python
from p01_load import load_data
```

The police data comes in several csv files with a folder for each Month-Year. Within each folder, there is a CSV file for each constabulary. We will concatenate these


```python
# Using a glob here to get all of the locations of the files we need
path = glob.glob(os.getcwd() + "/all_data/*/*.csv")
```


```python
police_data_df = load_data(file_locations=path, sqlcontext=sqlCtx)
```

    Loading CSV files to sqlcontext


#### Inspecting the data


```python
police_data_df.show(5)
```

    +--------------------+-------+--------------------+--------------------+---------+---------+--------------------+---------+--------------------+--------------------+---------------------+-------+
    |            Crime ID|  Month|         Reported by|        Falls within|Longitude| Latitude|            Location|LSOA code|           LSOA name|          Crime type|Last outcome category|Context|
    +--------------------+-------+--------------------+--------------------+---------+---------+--------------------+---------+--------------------+--------------------+---------------------+-------+
    |5ac3055389bf0d7f7...|2017-07|Metropolitan Poli...|Metropolitan Poli...| 0.763024|51.210698|On or near Wind H...|E01024033|        Ashford 002E|         Other theft| Investigation com...|   null|
    |db2ba97b31d87e77e...|2017-07|Metropolitan Poli...|Metropolitan Poli...| 1.185594|52.000179|On or near Main Road|E01029873|        Babergh 011B|        Public order| Investigation com...|   null|
    |                null|2017-07|Metropolitan Poli...|Metropolitan Poli...| 0.135866|51.587336|On or near Gibbfi...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |                null|2017-07|Metropolitan Poli...|Metropolitan Poli...| 0.140035|51.589112|On or near Beansl...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |                null|2017-07|Metropolitan Poli...|Metropolitan Poli...| 0.140035|51.589112|On or near Beansl...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    +--------------------+-------+--------------------+--------------------+---------+---------+--------------------+---------+--------------------+--------------------+---------------------+-------+
    only showing top 5 rows
    


Each dataset contains the following columns:


```python
police_data_df.printSchema()
```

    root
     |-- Crime ID: string (nullable = true)
     |-- Month: string (nullable = true)
     |-- Reported by: string (nullable = true)
     |-- Falls within: string (nullable = true)
     |-- Longitude: double (nullable = true)
     |-- Latitude: double (nullable = true)
     |-- Location: string (nullable = true)
     |-- LSOA code: string (nullable = true)
     |-- LSOA name: string (nullable = true)
     |-- Crime type: string (nullable = true)
     |-- Last outcome category: string (nullable = true)
     |-- Context: string (nullable = true)
    


The Data Dictionary is as follows


```python
dictionary = pd.read_csv('data_dictionary.csv')
```


```python
pd.set_option('display.max_colwidth', -1)
for elem in dictionary.to_records(index=False):
    print(elem[0] + ": " + elem[1])
```

    Reported by: The force that provided the data about the crime.
    Falls within: At present, also the force that provided the data about the crime. This is currently being looked into and is likely to change in the near future.
    Longitude and Latitude: The anonymised coordinates of the crime. See Location Anonymisation for more information.
    LSOA code and LSOA name: References to the Lower Layer Super Output Area that the anonymised point falls into, according to the LSOA boundaries provided by the Office for National Statistics.
    Crime type: One of the crime types listed in the Police.UK FAQ.
    Last outcome category: A reference to whichever of the outcomes associated with the crime occurred most recently. For example, this crime's 'Last outcome category' would be 'Formal action is not in the public interest'.
    Context: A field provided for forces to provide additional human-readable data about individual crimes. Currently, for newly added CSVs, this is always empty.


NOTE: LSOA (Lower Layer Super Output Area)

From NHS Data Dictionary (https://www.datadictionary.nhs.uk/data_dictionary/nhs_business_definitions/l/lower_layer_super_output_area_de.asp?shownav=1)
    
"<i>A Lower Layer Super Output Area (LSOA) is a GEOGRAPHIC AREA.
Lower Layer Super Output Areas are a geographic hierarchy designed to improve the reporting of small area statistics in England and Wales.
Lower Layer Super Output Areas are built from groups of contiguous Output Areas and have been automatically generated to be as consistent in population size as possible, and typically contain from four to six Output Areas. The Minimum population is 1000 and the mean is 1500.
There is a Lower Layer Super Output Area for each POSTCODE in England and Wales</i>"

How many Rows do we have?


```python
num_rows = police_data_df.count()
```


```python
num_rows
```




    18376429



## Cleaning the Data


```python
from p02_clean import clean_months, clean_location
```


```python
# The month column in the data is actually a Year-Month, here we will split that on the - delimiter and create a Year and Month_of_Year Column
police_data_clean = clean_months(police_data_df)
# Now lets create a Location and Town/City Column
police_data_clean = clean_location(police_data_clean)
```

    Cleaning Year and Month Columns...
    Creating Month_of_Year and Year columns...
    Cleaning Complete
    Cleaning Location and Town and City...
    Cleaning Complete



```python
police_data_clean.printSchema()
```

    root
     |-- Crime ID: string (nullable = true)
     |-- Month: string (nullable = true)
     |-- Reported by: string (nullable = true)
     |-- Falls within: string (nullable = true)
     |-- Longitude: double (nullable = true)
     |-- Latitude: double (nullable = true)
     |-- Location: string (nullable = true)
     |-- LSOA code: string (nullable = true)
     |-- LSOA name: string (nullable = true)
     |-- Crime type: string (nullable = true)
     |-- Last outcome category: string (nullable = true)
     |-- Context: string (nullable = true)
     |-- Year: integer (nullable = true)
     |-- Month_of_Year: string (nullable = true)
     |-- Town_City: string (nullable = true)
    


## Exploratory Analysis


```python
import pyspark.sql.functions as F
```

#### Are there any cases of when the constabulary that reported the crime is different to the constabulary area?


```python
police_data_df\
    .where(F.col("Reported by") != F.col("Falls within"))\
    .count()
```




    0



#### Number of Incidents over time


```python
crime_over_time = police_data_df\
                 .groupBy(["Month_of_Year", "Year"])\
                 .count()\
                 .toPandas()
```


```python
months = map(lambda x: calendar.month_abbr[x], range(1, 13))
crime_over_time["Month_of_Year"] = pd.Categorical(crime_over_time["Month_of_Year"], categories=months)
crime_time_series = crime_over_time.set_index(["Year", "Month_of_Year"]).sort_index().squeeze()
```


```python
plt.figure(figsize=(20, 5))
ax = crime_time_series.plot(kind = "line", color="b")
ax.set_xticks(range(0, len(crime_time_series.index)))
ax.set_xticklabels(list(crime_time_series.index))
crime_time_series.plot(rot=90)
plt.show()
```


![png](output_36_0.png)


There appears to be a linear trend and some periodicity with the numbers of reported crimes, although we do not have complete years in this dataset so we cannot truly infer this conclusion

#### Most Common Crime and Outcome Category Combination


```python
outcome_counts = police_data_df\
                 .groupBy(["Crime type", "Last outcome category"])\
                 .count()\
                 .toPandas()
```


```python
outcome_counts_series = outcome_counts\
.set_index(["Crime type", "Last outcome category"])\
.squeeze()\
.sort_values(ascending=False)\
.head(20)
```


```python
outcome_counts_series\
.sort_values(ascending=True)\
.plot(kind="barh", figsize=(10,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1158f2828>




![png](output_41_1.png)


#### The Most Common Type of Crime


```python
crime_type_counts = police_data_df\
                    .groupBy(police_data_df['Crime type'])\
                    .count()\
                    .sort(F.col("count").desc())\
                    .toPandas()
```


```python
crime_type_counts_series = crime_type_counts.set_index("Crime type").squeeze()
crime_type_counts_series = crime_type_counts_series*100/num_rows
```


```python
plt.figure(figsize=(10, 5))
crime_type_counts_series.plot(kind = "bar")
plt.show()
```


![png](output_45_0.png)


Anti-social behaviour makes up about 27% of crime in England - which is expected... It is concerning that violence and sexual offences is in second place

#### Which Town or City has the most crime?


```python
crime_town_city_counts = police_data_df\
                    .groupBy(police_data_df['Town_City'])\
                    .count()\
                    .sort(F.col("count").desc())\
                    .toPandas()
```


```python
plt.figure(figsize=(10,5))
sns.barplot(y=crime_town_city_counts.loc[1:20, "count"], x=crime_town_city_counts.loc[1:20, "Town_City"])
plt.xticks(rotation=90)
plt.show()
```


![png](output_49_0.png)


There appear to be a large amount of NAs....
Lets find out what the NAs are being caused by


```python
null_lsoa = police_data_df\
                    .where(police_data_df["LSOA name"].isNull())
```


```python
null_lsoa.count()/num_rows
```




    0.018709130049151552



So only about two percent of the data has no location information, we can exclude this

## Feature Engineering

Let's look at what the feature engineering code is actually doing


```bash
%%bash
cat p04_feature_engineer.py
```

    import pyspark.sql.functions as F
    
    def get_modelling_data(df):
        
        select_cols = ["Falls within", "Town_City", "Crime type", "Last outcome category", "Month_of_Year"]
        
        # Remove the crimes with no crime ID and no LSOA Information.
        # Then select the features of interest
        
        print('Filtering data with no Crime ID...')
        police_data_modelling = df\
            .filter(df["Crime ID"].isNotNull())\
            .select(select_cols)
        print('Filtering complete')
        return police_data_modelling
    
    def make_item_sets(df):
        # The FP growth algorithm (like association rules), needs the items to be concatenated into a list/array of "transactions".
        print('Making item sets...')
        print('Collapsing data to list of transactions')
        police_item_set = df.withColumn("items", F.array(df["Falls within"],
                                                         df["Town_City"],
                                                         df["Crime type"],
                                                         df["Last outcome category"],
                                                         df["Month_of_Year"]))
        # Select the items column and id
        print('Adding increasing id column...')
        
        police_item_set = police_item_set\
            .select("items")\
            .withColumn("id", F.monotonically_increasing_id())
        print('Itemset creation complete')
        
        return police_item_set
    
    
    def feature_engineer(df):
        """Invoke the full feature engineering pipeline"""
        print('Starting Feature Engineering pipeline...')
        selected_data = get_modelling_data(df)
        item_sets = make_item_sets(selected_data)
        print('Feature Engineering complet')
        return item_sets



```python
from p04_feature_engineer import *
```


```python
# Remove the crimes with no crime ID and no LSOA Information.
police_item_set = feature_engineer(police_data_clean)
```

    Starting Feature Engineering pipeline...
    Filtering data with no Crime ID...
    Filtering complete
    Making item sets...
    Collapsing data to list of transactions
    Adding increasing id column...
    Itemset creation complete
    Feature Engineering complet


The FP growth algorithm (like association rules), needs the items to be concatenated into a list/array of "transactions".


```python
police_item_set.show(truncate=False)
```

    +-------------------------------------------------------------------------------------------------------------------------------------+---+
    |items                                                                                                                                |id |
    +-------------------------------------------------------------------------------------------------------------------------------------+---+
    |[Metropolitan Police Service, Ashford, Other theft, Investigation complete; no suspect identified, Jul]                              |0  |
    |[Metropolitan Police Service, Babergh, Public order, Investigation complete; no suspect identified, Jul]                             |1  |
    |[Metropolitan Police Service, Barking and Dagenham, Bicycle theft, Status update unavailable, Jul]                                   |2  |
    |[Metropolitan Police Service, Barking and Dagenham, Burglary, Investigation complete; no suspect identified, Jul]                    |3  |
    |[Metropolitan Police Service, Barking and Dagenham, Burglary, Status update unavailable, Jul]                                        |4  |
    |[Metropolitan Police Service, Barking and Dagenham, Criminal damage and arson, Investigation complete; no suspect identified, Jul]   |5  |
    |[Metropolitan Police Service, Barking and Dagenham, Criminal damage and arson, Investigation complete; no suspect identified, Jul]   |6  |
    |[Metropolitan Police Service, Barking and Dagenham, Other theft, Status update unavailable, Jul]                                     |7  |
    |[Metropolitan Police Service, Barking and Dagenham, Other theft, Status update unavailable, Jul]                                     |8  |
    |[Metropolitan Police Service, Barking and Dagenham, Other theft, Status update unavailable, Jul]                                     |9  |
    |[Metropolitan Police Service, Barking and Dagenham, Public order, Investigation complete; no suspect identified, Jul]                |10 |
    |[Metropolitan Police Service, Barking and Dagenham, Public order, Status update unavailable, Jul]                                    |11 |
    |[Metropolitan Police Service, Barking and Dagenham, Public order, Investigation complete; no suspect identified, Jul]                |12 |
    |[Metropolitan Police Service, Barking and Dagenham, Public order, Investigation complete; no suspect identified, Jul]                |13 |
    |[Metropolitan Police Service, Barking and Dagenham, Robbery, Investigation complete; no suspect identified, Jul]                     |14 |
    |[Metropolitan Police Service, Barking and Dagenham, Robbery, Investigation complete; no suspect identified, Jul]                     |15 |
    |[Metropolitan Police Service, Barking and Dagenham, Robbery, Status update unavailable, Jul]                                         |16 |
    |[Metropolitan Police Service, Barking and Dagenham, Robbery, Investigation complete; no suspect identified, Jul]                     |17 |
    |[Metropolitan Police Service, Barking and Dagenham, Violence and sexual offences, Investigation complete; no suspect identified, Jul]|18 |
    |[Metropolitan Police Service, Barking and Dagenham, Violence and sexual offences, Investigation complete; no suspect identified, Jul]|19 |
    +-------------------------------------------------------------------------------------------------------------------------------------+---+
    only showing top 20 rows
    


## Modelling: Create the FP growth algorithm
For Association rules


```python
from p05_model import build_association_rule_model, extract_model_rules
```


```python
# Use a low support as we have a large dataset
model = build_association_rule_model(police_item_set, min_support=0.01, min_confidence=0.6)
```

    Fitting FPGrowth....
    Fit Complete


#### Extract the Association Rules


```python
rules_df_pd = extract_model_rules(model)
```

    Extracting Rules...
    Rule extraction complete
    Collecting Rules to Pandas...
    Collection Complete...



```python
rules_df_pd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Birmingham</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.605648</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Birmingham,West Midlands Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.606086</td>
    </tr>
    <tr>
      <th>2</th>
      <td>West Yorkshire Police,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.607850</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apr,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.617538</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mar,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.621105</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.621714</td>
    </tr>
    <tr>
      <th>6</th>
      <td>May,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.622566</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jul,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.622925</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jun,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.623030</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Criminal damage and arson</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.624007</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Other theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641287</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Manchester</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641456</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manchester,Greater Manchester Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641596</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Vehicle crime</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.698974</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Burglary</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.704225</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bicycle theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.720710</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Vehicle crime,Status update unavailable</td>
      <td>Metropolitan Police Service</td>
      <td>0.751311</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Birmingham</td>
      <td>West Midlands Police</td>
      <td>0.998351</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sheffield</td>
      <td>South Yorkshire Police</td>
      <td>0.998930</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Leeds</td>
      <td>West Yorkshire Police</td>
      <td>0.999067</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Birmingham,Investigation complete; no suspect ...</td>
      <td>West Midlands Police</td>
      <td>0.999073</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Westminster</td>
      <td>Metropolitan Police Service</td>
      <td>0.999508</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Bradford</td>
      <td>West Yorkshire Police</td>
      <td>0.999553</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Liverpool</td>
      <td>Merseyside Police</td>
      <td>0.999558</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Bristol</td>
      <td>Avon and Somerset Constabulary</td>
      <td>0.999614</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Manchester</td>
      <td>Greater Manchester Police</td>
      <td>0.999700</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Manchester,Investigation complete; no suspect ...</td>
      <td>Greater Manchester Police</td>
      <td>0.999919</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules_df_pd.to_csv('crime_associations.csv')
```


```python
# Stop the Spark Session
sc.stop()
```

## Rules Analysis
As you can see, the rules in the 98%+ confidence region appear to be rules that don't really tell us anything. i.e. Birmingham -> West Midlands Police. Let's remove those from the analysis


```python
useful_rules_df = rules_df_pd[rules_df_pd['confidence'] < 0.98]\
    .sort_values(by="confidence", ascending = False)
```


```python
useful_rules_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedent</th>
      <th>consequent</th>
      <th>confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>Vehicle crime,Status update unavailable</td>
      <td>Metropolitan Police Service</td>
      <td>0.751311</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bicycle theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.720710</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Burglary</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.704225</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Vehicle crime</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.698974</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manchester,Greater Manchester Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641596</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Manchester</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641456</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Other theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641287</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Criminal damage and arson</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.624007</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jun,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.623030</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jul,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.622925</td>
    </tr>
    <tr>
      <th>6</th>
      <td>May,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.622566</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.621714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mar,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.621105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apr,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.617538</td>
    </tr>
    <tr>
      <th>2</th>
      <td>West Yorkshire Police,Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.607850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Birmingham,West Midlands Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.606086</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Birmingham</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.605648</td>
    </tr>
  </tbody>
</table>
</div>



Now the rule with the highest confidence is **(Vehicle Crime + Status update unavailable -> Metropolitan Police Service)**. So what does this mean? This means that given that a crime is a vehicle crime and there is no status update, the probability that it falls within the constabulary of the Metropolitan Police is around 75% 

The other 3 rules in the 69%+ confidence/conditional probability region follow a similar pattern.
- **(Bicycle Theft -> Investigation complete; no suspect identified)**
- **(Burglary -> Investigation complete; no suspect identified)**
- **(Vehicle Crime -> Investigation complete; no suspect identified)**

So, it implies that the probability of no suspect being identified after a bike theft, burglary or vehicle crime is about 69-72%.

Another interesting rule is **(Manchester -> Investigation complete; no suspect identified)**. So what this is saying is, the model estimates that the probability that a reported crime leads to a complete investigation with no suspect identified, given that the crime occurred in Manchester around 64%

Another block of these rules is **(Unable to prosecute suspect -> Violence and sexual offences)**, which sounds worrying but doesn't really say much. The conditional probability of a crime being a violence and sexual offence, given that you were unable to prosecute the suspect is around 62%.


```python
sc.stop()
```
