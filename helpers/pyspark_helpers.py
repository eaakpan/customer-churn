import joblib
from config import PySparkConnection
from helpers.data_structures import  batch_prediction_cols
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, StringIndexer,\
OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def pyspark_batch_training_data_cleaner(df):
    train, test = df.randomSplit([0.7, 0.3])
    train.show(3)
    test.show(3)
    numerical_feat_cols = [c for c, t in df.dtypes if t.startswith('string') == False]
    string_feat_cols = [c for c, t in df.dtypes if t.startswith('string') == True]


    imputer = Imputer(inputCols=numerical_feat_cols,
                      outputCols=numerical_feat_cols)

    imputer = imputer.fit(train)
    train = imputer.transform(train)
    test = imputer.transform(test)

    numerical_vector_assembler = VectorAssembler(inputCols=numerical_feat_cols,
                                                 outputCol='numerical_feature_vector')

    train = numerical_vector_assembler.transform(train)
    test = numerical_vector_assembler.transform(test)

    scaler = StandardScaler(inputCol='numerical_feature_vector',
                            outputCol='scaled_numerical_feature_vector',
                            withStd=True, withMean=True)

    scaler = scaler.fit(train)

    train = scaler.transform(train)
    test = scaler.transform(test)

    indexer = StringIndexer(inputCols=string_feat_cols,
                            outputCols=[col + '_index' for col in string_feat_cols])

    indexer = indexer.fit(train)
    train = indexer.transform(train)
    test = indexer.transform(test)

    one_hot_encoder = OneHotEncoder(inputCols=[col + '_index' for col in string_feat_cols],
                                    outputCols=[col + '_index_one_hot'
                                                for col in string_feat_cols])

    one_hot_encoder = one_hot_encoder.fit(train)

    train = one_hot_encoder.transform(train)
    test = one_hot_encoder.transform(test)

    assembler = VectorAssembler(inputCols=['scaled_numerical_feature_vector'] +
                                          [col + '_index_one_hot' for col in string_feat_cols
                                           if 'churn' not in col],
                                outputCol='final_feature_vector')

    train = assembler.transform(train)
    test = assembler.transform(test)
    return train, test



def batch_model_training_pyspark():
    table = "churnset.customers"
    sc = PySparkConnection()


    df = sc.read(table)
    df = df.select(batch_prediction_cols)

    # # partitioning
    # df = sc.read(table).limit(12)
    # df = df.repartition(4, "customerid")
    # print(df.collect())


    train, test = pyspark_batch_training_data_cleaner(df)

    log_reg = LogisticRegression(featuresCol='final_feature_vector',
                                 labelCol='churn_index')


    log_reg = log_reg.fit(train)
    results = log_reg.transform(test)

    res = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='churn_index')




    ROC_AUC = res.evaluate(results)
    print("Logistic Regression accuracy is :", ROC_AUC)

    # joblib.dump(log_reg, 'runtime_data/for_models/pyspark_lr_model.joblib')
    # log_reg.save('runtime_data/for_models/pyspark_lr_model')
    # return print("batch model has been completed and saved")