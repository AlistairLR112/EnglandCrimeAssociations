def load_data(file_locations, sqlcontext):
    print("Loading CSV files to sqlcontext...")
    police_data_df = sqlcontext.read.format("csv")\
                .option("header", "true")\
                .option("inferSchema", "true")\
                .load(file_locations)
    print('Load Complete')
    return police_data_df
                