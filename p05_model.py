from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import concat_ws, col
import pandas as pd

def build_association_rule_model(item_set, min_support, min_confidence):
    # Use a low support as we have a large dataset
    fp_growth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
    
    print('Fitting FPGrowth....')
    model = fp_growth.fit(item_set)
    print('Fit Complete')
    return model

def extract_model_rules(model):
    # Extract the association rules from the model
    print('Extracting Rules...')
    rules = model.associationRules
    print('Rule extraction complete')
    
    # Minor tidying to save as pandas dataframe. Antecedent and Consequent are array types
    # Concatenate into one string and sort by highest confidence
    
    print('Collecting Rules to Pandas...')
    rules_df = rules\
            .withColumn("antecedent" ,concat_ws(",", rules["antecedent"]))\
            .withColumn("consequent" ,concat_ws(",", rules["consequent"]))\
            .sort(col("confidence"))\
            .toPandas()
    print('Collection Complete...')
    
    return rules_df