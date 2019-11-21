from pyspark.sql.types import ArrayType, StringType
import pyspark.sql.functions as F

clear_duplicates = F.udf(lambda x: list(set(x)), ArrayType(StringType()))

def get_modelling_data(df):
    
    select_cols = ["Falls within", "Town_City", "Crime type", "Last outcome category", "Month_of_Year"]
    
    # Remove the crimes with no crime ID and no LSOA Information.
    # Then select the features of interest
    
    print('Filtering data with no Crime ID and no outcome category..')
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
        .select("items")
    print('Itemset creation complete')
    
    return police_item_set


def feature_engineer(df):
    """Invoke the full feature engineering pipeline"""
    print('Starting Feature Engineering pipeline...')
    selected_data = get_modelling_data(df)
    item_sets = make_item_sets(selected_data)
    print('Feature Engineering complet')
    return item_sets
