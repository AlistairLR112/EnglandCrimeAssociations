
# Association Rule Mining for reported street crimes in England & Wales
The aim here is to see if there are any associations between the reported aspects of street crime, such as Month of Year, Location, Crime type etc.
This will be done in Pyspark due to the size of the data but it will still be possible to execute on a local cluster.

The data can be downloaded from here: https://data.police.uk/data/.

The date range for this data is December 2010 - July 2019 and all constabularies in England & Wales were selected (we will be excluding British Transport Police and Police Service of Northern Ireland)

## What is an Association Rule?
*Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness. This rule-based approach also generates new rules as it analyzes more data. The ultimate goal, assuming a large enough dataset, is to help a machine mimic the human brain’s feature extraction and abstract association capabilities from new uncategorized data.*

We will be looking for rules with a high level of confidence

*Confidence is an indication of how often the rule has been found to be true... Confidence can be interpreted as an estimate of the conditional probability*


```python
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
```


```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


#### Set up Spark
Running Spark locally using 6 out of 8 cores


```python
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
```


```python
spark = SparkSession.builder\
        .master("local[7]")\
        .appName("Crime Assocations")\
        .config("spark.executor.memory", "6g")\
        .config("spark.memory.fraction", 0.7)\
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
path = glob.glob(os.getcwd() + "/all_data/*/*-street.csv")
```


```python
police_data_df = load_data(file_locations=path, sqlcontext=sqlCtx)
```

    Loading CSV files to sqlcontext...
    Load Complete


#### Inspecting the data


```python
police_data_df.select(police_data_df.columns[1:]).show()
```

    +-------+--------------------+--------------------+---------+---------+--------------------+---------+--------------------+--------------------+---------------------+-------+
    |  Month|         Reported by|        Falls within|Longitude| Latitude|            Location|LSOA code|           LSOA name|          Crime type|Last outcome category|Context|
    +-------+--------------------+--------------------+---------+---------+--------------------+---------+--------------------+--------------------+---------------------+-------+
    |2012-08|Metropolitan Poli...|Metropolitan Poli...|-0.508053|50.809718|On or near Claigm...|E01031464|           Arun 007F|       Violent crime|  Under investigation|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| -1.01393|51.899297|On or near St Mic...|E01017673| Aylesbury Vale 010C|         Other crime|  Under investigation|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.964612|52.045416|On or near Barnes...|E01029896|        Babergh 004E|       Violent crime|  Under investigation|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.134947|51.588063|On or near Mead G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.140634|51.583427|On or near Rams G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.134947|51.588063|On or near Mead G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.140634|51.583427|On or near Rams G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.134947|51.588063|On or near Mead G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.145888|51.593835|On or near Provid...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.141143|51.590873|On or near Furze ...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.140634|51.583427|On or near Rams G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.140634|51.583427|On or near Rams G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.134947|51.588063|On or near Mead G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.134947|51.588063|On or near Mead G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.140035|51.589112|On or near Beansl...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.134947|51.588063|On or near Mead G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.137065|51.583672|On or near Police...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.137065|51.583672|On or near Police...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.135866|51.587336|On or near Gibbfi...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    |2012-08|Metropolitan Poli...|Metropolitan Poli...| 0.134947|51.588063|On or near Mead G...|E01000027|Barking and Dagen...|Anti-social behav...|                 null|   null|
    +-------+--------------------+--------------------+---------+---------+--------------------+---------+--------------------+--------------------+---------------------+-------+
    only showing top 20 rows
    


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




    52835178



## Cleaning the Data


```python
from p02_clean import clean_months, clean_location, clean_non_england
```


```python
# The month column in the data is actually a Year-Month, here we will split that on the - delimiter and create a Year and Month_of_Year Column
police_data_clean = clean_months(police_data_df)
# Now lets create a Location and Town/City Column
police_data_clean = clean_location(police_data_clean)
police_data_clean = clean_non_england(police_data_clean)
```

    Cleaning Year and Month Columns...
    Creating Month_of_Year and Year columns...
    Cleaning Complete
    Cleaning Location and Town and City...
    Cleaning Complete
    Removing non England and Wales entries
    Removal Complete



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


```python
import p03_eda as eda
```

#### Are there any cases of when the constabulary that reported the crime is different to the constabulary area?


```python
police_data_clean\
    .where(F.col("Reported by") != F.col("Falls within"))\
    .count()
```




    0



#### Number of Incidents over time


```python
plt.figure(figsize=(20, 5))
crime_over_time_plot = eda.plot_crime_time_series(police_data_clean, read=True)
plt.show()
```

    Setting Month to a categorical variable...
    Setting complete
    Converting to Series object...
    Conversion complete
    Creating plot object
    Complete.. Plotting...



![png](output_35_1.png)


There appears to be a stationary trend with some periodicity with the numbers of reported crimes, although we do not have complete years in this dataset. It looks like there is a pattern to the level/numbers of street crimes!

#### Most Common Crime and Outcome Category Combination


```python
plt.figure(figsize=(10, 10))
outcome_category_plot = eda.plot_crime_type_and_category_counts(police_data_clean, read=True)
plt.show()
```

    Converting to Series
    Collapsing Multi Index
    Plotting...



![png](output_38_1.png)


It seems the most common type to outcome association is an anti social behaviour crime with no recorded outcome

#### The Most Common Type of Crime


```python
plt.figure(figsize=(10, 5))
crime_type_plot = plot_crime_counts(police_data_clean, read=True)
plt.show()
```

    Converting to Series object...
    Plotting...



![png](output_41_1.png)


Anti-social behaviour makes up about 35% of crime in England - which is expected... It is concerning that violence and sexual offences is in second place

#### Which Town or City has the most crime?


```python
plt.figure(figsize=(10,5))
crime_town_city_counts = plot_crime_town_city_counts(police_data_clean, read=True)
plt.xticks(rotation=90)
plt.show()
```

    Converting to Series
    Plotting...



![png](output_44_1.png)


## Feature Engineering

Let's look at what the feature engineering code is actually doing


```bash
%%bash
cat p04_feature_engineer.py
```

    import pyspark.sql.functions as F
    from pyspark.sql.types import ArrayType, StringType
    
    clear_duplicates = F.udf(lambda x: list(set(x)), ArrayType(StringType()))
    
    def get_modelling_data(df):
        
        select_cols = ["Falls within", "Town_City", "Crime type", "Last outcome category", "Month_of_Year"]
        
        # Remove the crimes with no crime ID and no LSOA Information.
        # Then select the features of interest
        
        print('Filtering data with no Crime ID...')
        police_data_modelling = df\
            .filter(df["Crime ID"].isNotNull() & df["Last outcome category"].isNotNull())\
            .select(select_cols)
        print('Filtering complete')
        return police_data_modelling
    
    def make_item_sets(df):
        # The FP growth algorithm (like association rules), needs the items to be concatenated into a list/array of "transactions".
        print('Making item sets...')
        print('Collapsing data to list of transactions')
        police_item_set = df.withColumn("items_temp", F.array(df["Falls within"],
                                                         df["Town_City"],
                                                         df["Crime type"],
                                                         df["Last outcome category"],
                                                         df["Month_of_Year"]))
        
        police_item_set = police_item_set.withColumn("items", clear_duplicates(police_item_set["items_temp"]))
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
    Filtering data with no Crime ID and no outcome category..
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

    +----------------------------------------------------------------------------------------------------------------------------------+
    |items                                                                                                                             |
    +----------------------------------------------------------------------------------------------------------------------------------+
    |[Metropolitan Police Service, Aug, Arun, Violent crime, Under investigation]                                                      |
    |[Other crime, Metropolitan Police Service, Aug, Aylesbury Vale, Under investigation]                                              |
    |[Metropolitan Police Service, Aug, Violent crime, Under investigation, Babergh]                                                   |
    |[Burglary, Barking and Dagenham, Metropolitan Police Service, Aug, Under investigation]                                           |
    |[Barking and Dagenham, Metropolitan Police Service, Investigation complete; no suspect identified, Aug, Criminal damage and arson]|
    |[Offender given a drugs possession warning, Drugs, Barking and Dagenham, Metropolitan Police Service, Aug]                        |
    |[Barking and Dagenham, Metropolitan Police Service, Aug, Vehicle crime, Under investigation]                                      |
    |[Barking and Dagenham, Metropolitan Police Service, Aug, Vehicle crime, Under investigation]                                      |
    |[Barking and Dagenham, Metropolitan Police Service, Aug, Vehicle crime, Under investigation]                                      |
    |[Barking and Dagenham, Metropolitan Police Service, Investigation complete; no suspect identified, Aug, Violent crime]            |
    |[Offender sent to prison, Barking and Dagenham, Metropolitan Police Service, Aug, Violent crime]                                  |
    |[Barking and Dagenham, Metropolitan Police Service, Aug, Violent crime, Under investigation]                                      |
    |[Offender given a caution, Barking and Dagenham, Metropolitan Police Service, Aug, Violent crime]                                 |
    |[Barking and Dagenham, Metropolitan Police Service, Aug, Violent crime, Under investigation]                                      |
    |[Barking and Dagenham, Metropolitan Police Service, Aug, Violent crime, Under investigation]                                      |
    |[Burglary, Barking and Dagenham, Metropolitan Police Service, Aug, Under investigation]                                           |
    |[Burglary, Barking and Dagenham, Metropolitan Police Service, Aug, Under investigation]                                           |
    |[Burglary, Barking and Dagenham, Metropolitan Police Service, Aug, Under investigation]                                           |
    |[Burglary, Barking and Dagenham, Metropolitan Police Service, Aug, Under investigation]                                           |
    |[Barking and Dagenham, Metropolitan Police Service, Aug, Vehicle crime, Under investigation]                                      |
    +----------------------------------------------------------------------------------------------------------------------------------+
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
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Theft from the person</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.601021</td>
      <td>1.274382</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Greater Manchester Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.603717</td>
      <td>1.280098</td>
    </tr>
    <tr>
      <td>2</td>
      <td>West Midlands Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.606495</td>
      <td>1.285988</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Birmingham</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.607874</td>
      <td>1.288914</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Birmingham,West Midlands Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.608291</td>
      <td>1.289796</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.621860</td>
      <td>2.453091</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Criminal damage and arson</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641662</td>
      <td>1.360556</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Manchester</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.650860</td>
      <td>1.380060</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Manchester,Greater Manchester Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.651027</td>
      <td>1.380413</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Other theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.666711</td>
      <td>1.413669</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Vehicle crime</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.692018</td>
      <td>1.467329</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Burglary</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.711254</td>
      <td>1.508117</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Bicycle theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.719270</td>
      <td>1.525113</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Birmingham</td>
      <td>West Midlands Police</td>
      <td>0.998434</td>
      <td>20.240328</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Birmingham,Investigation complete; no suspect ...</td>
      <td>West Midlands Police</td>
      <td>0.999117</td>
      <td>20.254189</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Sheffield</td>
      <td>South Yorkshire Police</td>
      <td>0.999125</td>
      <td>35.791218</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Leeds</td>
      <td>West Yorkshire Police</td>
      <td>0.999156</td>
      <td>18.917528</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Westminster</td>
      <td>Metropolitan Police Service</td>
      <td>0.999549</td>
      <td>5.338211</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Bradford</td>
      <td>West Yorkshire Police</td>
      <td>0.999584</td>
      <td>18.925635</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Liverpool</td>
      <td>Merseyside Police</td>
      <td>0.999654</td>
      <td>38.381717</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Manchester</td>
      <td>Greater Manchester Police</td>
      <td>0.999672</td>
      <td>16.812479</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Bristol</td>
      <td>Avon and Somerset Constabulary</td>
      <td>0.999688</td>
      <td>34.073433</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Manchester,Investigation complete; no suspect ...</td>
      <td>Greater Manchester Police</td>
      <td>0.999928</td>
      <td>16.816784</td>
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
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>12</td>
      <td>Bicycle theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.719270</td>
      <td>1.525113</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Burglary</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.711254</td>
      <td>1.508117</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Vehicle crime</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.692018</td>
      <td>1.467329</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Other theft</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.666711</td>
      <td>1.413669</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Manchester,Greater Manchester Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.651027</td>
      <td>1.380413</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Manchester</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.650860</td>
      <td>1.380060</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Criminal damage and arson</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.641662</td>
      <td>1.360556</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Unable to prosecute suspect</td>
      <td>Violence and sexual offences</td>
      <td>0.621860</td>
      <td>2.453091</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Birmingham,West Midlands Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.608291</td>
      <td>1.289796</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Birmingham</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.607874</td>
      <td>1.288914</td>
    </tr>
    <tr>
      <td>2</td>
      <td>West Midlands Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.606495</td>
      <td>1.285988</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Greater Manchester Police</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.603717</td>
      <td>1.280098</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Theft from the person</td>
      <td>Investigation complete; no suspect identified</td>
      <td>0.601021</td>
      <td>1.274382</td>
    </tr>
  </tbody>
</table>
</div>



Now the rule with the highest confidence is **(Bicycle Theft -> Investigation complete; no suspect identified)**. So what does this mean? This means that given that a crime is a Bike Theft, the probability the investigation will be complete with no suspect identified is around 72%

The other 3 rules in the 65%+ confidence/conditional probability region follow a similar pattern.
- **(Other theft -> Investigation complete; no suspect identified)**
- **(Burglary -> Investigation complete; no suspect identified)**
- **(Vehicle Crime -> Investigation complete; no suspect identified)**

So, it implies that the probability of no suspect being identified after a burglary, vehicle crime, an incident of criminal damage or arr is about 69-72%.

Another interesting rule is **(Manchester -> Investigation complete; no suspect identified)**. So what this is saying is, the model estimates that the probability that a reported crime leads to a complete investigation with no suspect identified, given that the crime occurred in Manchester around 65%

Another block of these rules is **(Unable to prosecute suspect -> Violence and sexual offences)**, which sounds worrying but doesn't really say much. The conditional probability of a crime being a violence and sexual offence, given that you were unable to prosecute the suspect is around 61%.


```python
sc.stop()
```
