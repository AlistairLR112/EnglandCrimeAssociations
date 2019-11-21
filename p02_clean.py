import pyspark.sql.functions as F
import calendar

# In order to convert the month from a number representing the month index, this function has been created to map that index
# to its 3 letter abbreviation
get_month_name = F.udf(lambda x: calendar.month_abbr[x])

def clean_months(df):
    print('Cleaning Year and Month Columns...')
    
    # Split the month column into a a Month number and a Year
    month_year_split = F.split(df['Month'], '-')
    
    # Assign the Year and Month Number as new columns
    df = df.withColumn('Year', month_year_split.getItem(0))\
                    .withColumn('Month_of_Year', month_year_split.getItem(1))
    
    print('Creating Month_of_Year and Year columns...')
    # Ensure that both of the new columns created are integer and not string types
    df = df.withColumn("Month_of_Year", df.Month_of_Year.cast("int"))\
                    .withColumn("Year", df.Year.cast("int"))
    
    # Ensure the month is a name of a month
    df_with_month = df.withColumn("Month_of_Year", get_month_name(df["Month_of_Year"]))
    
    print('Cleaning Complete')

    return df_with_month


def clean_location(df):
    # Remove the On or near part of the string in Location as it adds no information
    #' Remove the code at the end of LSOA name to get the Town/City
    print('Cleaning Location and Town and City...')
    df_with_location = df.withColumn('Location', F.regexp_replace('Location', 'On or near ', ''))\
                        .withColumn("Town_City", F.regexp_replace('LSOA name', ' [0-9]{3}\w', ''))
    print('Cleaning Complete')
    return df_with_location

def clean_non_england(df):
    print('Removing non England and Wales entries')
    df_england = df.where((df["Falls within"] != "Police Service of Northern Ireland") & (df["Falls within"] != "British Transport Police"))
    print('Removal Complete')
    return df_england

    
    