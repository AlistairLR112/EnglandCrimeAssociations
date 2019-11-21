import calendar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotnine import ggplot, geoms
import pyspark.sql.functions as F

sns.set_style('darkgrid')
colours = sns.color_palette("PuBu", 10)

def plot_crime_time_series(df, read=False):
    """
    
    Plots the number of crimes that occured each month'
    
    """
    
    if read:
        crime_over_time = pd.read_csv("./eda_data/crime_over_time.csv", 
                                      usecols = ["Month_of_Year", "Year", "count"])
    else:
        print('Grouping by Month and Year and collecting to Pandas...')
        crime_over_time = df\
                         .groupBy(["Month_of_Year", "Year"])\
                         .count()\
                         .toPandas()
        print('Collected to Pandas')
        crime_over_time.to_csv("./eda_data/crime_over_time.csv")
    
    
    print('Setting Month to a categorical variable...')
    months = map(lambda x: calendar.month_abbr[x], range(1, 13))
    crime_over_time["Month_of_Year"] = pd.Categorical(crime_over_time["Month_of_Year"], categories=months)
    print('Setting complete')
    
    print('Converting to Series object...')
    crime_time_series = crime_over_time\
                        .set_index(["Year", "Month_of_Year"])\
                        .sort_index()\
                        .squeeze()
    print('Conversion complete')
    
    print('Creating plot object')
    plot = crime_time_series.plot(kind = "line", color="b", title = "Incidents of Reported Street Crime (Dec 2010 - Jul 2019)")
    
    plot.set_xticks(range(0, len(crime_time_series.index)))
    plot.set_xticklabels(list(crime_time_series.index), rotation=90)
    print("Complete.. Plotting...")

    return plot

def plot_crime_counts(df, read=False):
    
    if read:
        crime_type_counts = pd.read_csv("./eda_data/crime_type_counts.csv", usecols=['Crime type', 'count'])
    
    else:
        print('Grouping by Crime type and collecting to Pandas..')
        crime_type_counts = df\
                        .groupBy(df['Crime type'])\
                        .count()\
                        .sort(F.col("count").desc())\
                        .toPandas()
        crime_type_counts.to_csv("./eda_data/crime_type_counts.csv", header=True)
    
    print('Converting to Series object...')
    crime_type_counts_series = crime_type_counts\
                               .set_index("Crime type")\
                               .squeeze()\
                               .apply(lambda x: x*100/sum(crime_type_counts["count"]))
    
    print('Plotting...')
    plot = sns.barplot(x = crime_type_counts_series.values, 
                       y = crime_type_counts_series.index,
                       color='b')
    
    return plot
    
    
def plot_crime_type_and_category_counts(df, read=False):
    
    if read:
        outcome_counts = pd.read_csv("./eda_data/outcome_counts.csv", 
                                     usecols=["Crime type", "Last outcome category", "count"])
    else:
        print('Grouping by Crime type and Outcome Category')
        outcome_counts = df\
                     .groupBy(["Crime type", "Last outcome category"])\
                     .count()\
                     .sort(F.col("count").desc())\
                     .toPandas()
        outcome_counts.to_csv("./eda_data/outcome_counts.csv", header=True)
    
    print('Converting to Series')
    outcome_counts_series = outcome_counts\
                            .set_index(["Crime type", "Last outcome category"])\
                            .squeeze()\
                            .apply(lambda x: x*100/sum(outcome_counts["count"]))\
                            .head(20)
    print('Collapsing Multi Index')
    index = [str(x) + " -> " + str(y) for x, y in outcome_counts_series.index]
    
    print("Plotting...")
    plot = sns.barplot(x = outcome_counts_series.values, y = index, color='b')
        
    plot.set_title('% Total Reported Crime Type and Outcome combination (Dec 2010 - Jul 2019)')
    plot.set_xlabel('% of Reported Crimes') 
    plot.set_ylabel('Crime Type -> Outcome')
    
    return plot

def plot_crime_town_city_counts(df, read=False):
    
    if read:
        crime_town_city_counts = pd.read_csv(
            "./eda_data/crime_town_city_counts.csv", 
            usecols=["Town_City", "count"]
        )        
    else:
        print('Grouping by Town or City')
        crime_town_city_counts = df\
                        .groupBy(df['Town_City'])\
                        .count()\
                        .sort(F.col("count").desc())\
                        .toPandas()
        crime_town_city_counts.to_csv("./eda_data/crime_town_city_counts.csv", header=True)
    
    print('Converting to Series')
    crime_town_city_counts_series = crime_town_city_counts\
                                    .set_index(["Town_City"])\
                                    .squeeze()\
                                    .head(20)
    print('Plotting...')
    plot = sns.barplot(y=crime_town_city_counts_series.index, 
                       x=crime_town_city_counts_series.values,
                       color='b')
    
    plot = plot.set_xticklabels(np.arange(0, max(crime_town_city_counts_series.values), 10000))    
    return plot