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
