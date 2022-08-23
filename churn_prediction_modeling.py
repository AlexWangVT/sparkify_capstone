# import libraries
import os
from typing import final
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1' # set environment

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import Row, Window
import builtins as p
from pyspark.sql.functions import col, udf, sum, when, avg, stddev, max
from pyspark.sql.types import IntegerType, StringType
import matplotlib.pyplot as plt
import pyspark.pandas as ps
from sklearn.metrics import f1_score
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from sklearn.inspection import permutation_importance
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, MultilayerPerceptronClassifier
import datetime
import time
from IPython.display import display


# create a spark session
spark = SparkSession.builder.appName('Sparkify').getOrCreate()


def load_data(data_path):
    '''
    Load raw data
    data_path = 'mini_sparkify_event_data.json'
    '''
    df = spark.read.json(data_path)
    return df


def clean_data(df):
    '''
    This function is used for data clean, including removing null users, cerating churn column, time data processing,
    userAgent data processing
    '''
    # remove events without userId
    df = df.filter(df.userId != '')
    
    # define churn and add churn column
    cancellation_event = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0, IntegerType())
    df = df.withColumn('churn', cancellation_event('page'))
    window = Window.partitionBy('userId').rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    df = df.withColumn('churn', sum('churn').over(window))
    # process time data and create columnds for hour, day, month, weekday
    get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).hour)
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).day)
    get_month = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).month)
    get_weekday = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime('%w'))
    funcs = {'hour': get_hour,'day':get_day, 'month':get_month, 'week_day':get_weekday }
    for label, func in funcs.items():
        df = df.withColumn(label, func(df.ts))
        print(f'Column {label} add successfully!')
    # process user agent
    sys_general = {'Compatible': 'Windows', 'Ipad': 'iOS', 'Iphone': 'iOS', 'Macintosh': 'Mac', 
                'Windows nt 5.1': 'Windows','Windows nt 6.0': 'Windows', 'Windows nt 6.1': 'Windows',
                'Windows nt 6.2': 'Windows', 'Windows nt 6.3': 'Windows', 'X11': 'Linux'}
    # get the strings in the parenthesis
    get_sys = udf(lambda x: sys_general[x[x.find('(') + 1:x.find(')')].split(';')[0].capitalize()])
    df = df.withColumn('computer_systems', get_sys(df.userAgent))
    # get state
    get_state = udf(lambda x: x.split(',')[1].strip())
    clean_df = df.withColumn('State', get_state(df.location))

    return clean_df


def prepare_ml_data(df, data_save):
    '''
    The function converts categorical features to dummy variables. Categorical features include gender, level, 
    userAgent (sys_list), location (state), page event. The function also calculates the number of artists users like, 
    session duration, number of sessions, supscription age, and finally joins all features together into one dataset 
    for the modeling purpose.

    The features include gender_df, level_df, sys_df, loc_df, user_page_distribution, artists, user_session_hr, 
    session_count, reg_df.
    '''
    # get dummy variables
    def get_dummy(df, column, val):
        '''
        convert categorical features to dummy variables.
        The get_dummy func is used to convert 'gender' and 'level' columns to dummy.
        '''
        col_df = df.select('userId', column).dropDuplicates()
        col_df = col_df.withColumn(f'{column}_num', when(col(column) == val, 1).otherwise(0)).select('userId', col(f'{column}_num').cast('int'))
        return col_df
    # run get_dummy func to convert gender to dummy feature
    gender_df = get_dummy(df, 'gender', 'M')
    # run get_dummy func to convert level to dummy feature
    level_df = get_dummy(df, 'level', 'paid')
    
    # get dummy for state and userAgent
    def dummy_sys_state(df, column_name):
        '''
        We use this function to convert useragent and state to dummy variables.
        sys_list - the list of unique values
        list_exp - convert categorical features to 0-1 dummy variable
        '''
        sys_list = df.select(column_name).distinct().rdd.flatMap(lambda x: x).collect()
        list_exp = [when(col(column_name) == x, 1).otherwise(0).alias(x) for x in sys_list]
        df = df.select('userId', *list_exp).dropDuplicates()
        return df

    # Create dummy variables for userAgent
    sys_df = dummy_sys_state(df, 'computer_systems')
    # Create dummy variables for state
    loc_df = dummy_sys_state(df, 'State')

    # To get the Number of ThumbsUp, ThumbsDown, InviteFriends, downgrades, ...
    # I think it is better to get all actions except Chorn actions (Cancel, cancelation confirmation)
    # The to normalize them as percent to sum all to 100.
    # pivot the column 'page' by useId
    user_page_distribution = df.groupby('userId').pivot('page').count().na.fill(0) 
    # Drop Cancel	Cancellation Confirmation columns
    drop_col = ['Cancel', 'Cancellation Confirmation']
    user_page_distribution = user_page_distribution.drop(*drop_col)

    # Normalize each row sum to 1
    # the columns to be summed and add a total column
    page_cols = user_page_distribution.columns[1:]
    old_col_names = user_page_distribution.columns
    user_page_distribution = user_page_distribution.withColumn('rowsum', p.sum([col(c) for c in page_cols]))
    # Apply normalization per column
    for c in page_cols:
        user_page_distribution = user_page_distribution.withColumn(f'{c}_norm', col(c)/col('rowsum')*100)
    # Remove the total column 
    user_page_distribution = user_page_distribution.drop(*page_cols)
    user_page_distribution = user_page_distribution.drop('rowsum')
    # Rename the normalized columns back
    new_col_names = user_page_distribution.columns
    for idx in range(len(old_col_names)):
        user_page_distribution = user_page_distribution.withColumnRenamed(new_col_names[idx], old_col_names[idx])

    # get number of artists per user
    artists = df.select('userId', 'artist').dropDuplicates().groupBy('userId').count().withColumnRenamed('count', 'num_artist')
    
    # get session duration of each user
    start = df.groupBy('userId', 'sessionId').min('ts').withColumnRenamed('min(ts)', 'start')
    end = df.groupBy('userId', 'sessionId').max('ts').withColumnRenamed('max(ts)', 'end')
    sesseion_duration = start.join(end, ['userId', 'sessionId'])
    ticks_per_hours = 1000 * 60 * 60
    session_df = sesseion_duration.select('userId', 'sessionId', ((col('end')-col('start'))/ticks_per_hours).alias('session_duration (h)'))
    user_session_hr = session_df.groupBy('userId').agg(avg('session_duration (h)').alias('mean_duration'), stddev('session_duration (h)').alias('std_duration')).na.fill(0)
    
    # get number of sessions per user
    session_count = df.select('userId', 'sessionId').dropDuplicates().groupBy('userId').count().withColumnRenamed('count', 'num_sessions')

    # get subscription duration for each user
    def subscription_age(df, col_name):
        '''
        This function is used to calculate the duration of each user being active.
        '''
        # timestamp of users registration
        reg_time = df.select('userId', 'registration').dropDuplicates().withColumnRenamed('registration', 'registration_time')
        # The maximum timestamp found for the user
        max_time = df.groupBy('userId').max('ts').withColumnRenamed('max(ts)', 'current')
        reg_df = reg_time.join(max_time, 'userId')
        ticks_per_day = 1000 * 60 * 60 * 24 # as the timestamp is in ticks (0.001 seconds)
        # Merge in one df
        reg_df = reg_df.select('userId', ((reg_df.current - reg_df.registration_time)/ticks_per_day).alias(col_name))
        return reg_df
    # create a subscription_age column
    reg_df = subscription_age(df, 'subscription_age')

    # join all features into one table for modeling 
    features = [gender_df, level_df, sys_df, loc_df, user_page_distribution, artists, user_session_hr, session_count, reg_df]
    # initialize the joined table
    churn_users = df.select('userId', 'churn').dropDuplicates()
    new_df = churn_users
    # join each feature to the modeling dataset
    for i, feature in enumerate(features):
        new_df = new_df.join(feature, 'userId', how = 'inner')
        print(f"the new frame's dimensions is: {new_df.count()} * {len(new_df.columns)}")

    final_df = new_df.orderBy('userId', ascending = True)
    print('*** Dataset is ready! ***')

    # save the final dataset locally
    try:
        final_df.write.csv(data_save, header = True)
    except:
        pass
    return final_df


def load_clean_transfer(data_path, save_data_as):
    '''
    This is the data pipeline that loads, clean, and transform the raw data.
    data_path - 'mini_sparkify_event_data.json'
    save_data_as - the path where we save the processed data 
    '''
    df = load_data(data_path)
    clean_df = clean_data(df)
    prepare_ml_data(clean_df, save_data_as)


def load_transform_ml_data(path):
    '''
    This function loads the processed data, change data type from string to integer or float, 
    assemble features into one list, and scale features by standard scaling method.
    '''
    # load data
    final_df = spark.read.csv(path, header=True)
    # replace space in column names with underscore
    for col in final_df.columns:
        final_df = final_df.withColumnRenamed(col, col.replace(' ', '_').replace('-', '_'))
    
    # change string userId to integer or float type
    final_df = final_df.withColumn('userId', final_df.userId.cast(IntegerType()))
    final_df = final_df.withColumn('churn', final_df.churn.cast(IntegerType()))
    for col in final_df.columns[2:80]:
        final_df = final_df.withColumn(col, final_df[col].cast(IntegerType()))

    for col in final_df.columns[80:]:
        final_df = final_df.withColumn(col, final_df[col].cast(FloatType()))
    
    # Replace nulls with 0
    final_df = final_df.na.fill(0)

    # Collect features using VectorAssembler
    feature_labels = final_df.columns[2:]
    assembler = VectorAssembler(inputCols=feature_labels, outputCol='features')
    input = assembler.transform(final_df)

    # Scale features
    feature_scaler = StandardScaler(withMean=True, withStd=True, inputCol='features', outputCol='scaled_features')
    scaler_fit = feature_scaler.fit(input)
    scaled_input = scaler_fit.transform(input)
    model_data = scaled_input.select(scaled_input.churn.alias('label'), scaled_input.scaled_features.alias('features'))

    # split the dataset into training and testing sets with 90% of the data used for training and 10% for testing
    train, test = model_data.randomSplit([0.9, 0.1], seed = 42)
    return train, test, feature_labels


def model_fit(data, model_type, param_grid, save_model, num_folds=10, random_seed=42):
    '''
    This function is used to fit the defined model.
    cross_val - model cross validation
    model_type - the type of model we used which includ logistic regression, decision tree, gradient boosting, random forest, and multiperception layer
    param_grid - the search window of hyperparameters
    evaluator - the type of evaluation metrics
    numFolds - the number of cross validation folds
    '''
    # define cross validation estimator
    cross_val = CrossValidator(estimator=model_type, estimatorParamMaps=param_grid, \
                            evaluator=MulticlassClassificationEvaluator(), numFolds=num_folds, seed=random_seed)
    
    start = time.time()
    # fit model
    model = cross_val.fit(data)      
    training_duration = time.time() - start
    print("training time is {}s".format(training_duration))
    # save the model, overwrite the model if already exists
    try:
        model.save(save_model)  
    except:
        model.write().overwrite().save(save_model)

    return model    


def get_metrics(metrics):
    '''
    This function is used to get metrics of training model.
    acc - model accuracy
    general_score - the general precision, recall, and f1 score for all users (churn = 0 or 1)
    label_score - the score for active or churn users respectively
    '''
    acc = metrics.accuracy
    general_score = np.array((metrics.weightedFMeasure(), metrics.weightedPrecision, metrics.weightedRecall,
                              metrics.weightedTruePositiveRate, metrics.weightedFalsePositiveRate))
    general_score = general_score.reshape(1, general_score.shape[0])
    labels = ['General'] + [f'Churn={x}' for x in metrics.labels]
    label_score = np.array((metrics.fMeasureByLabel(), metrics.precisionByLabel, metrics.recallByLabel,
                            metrics.truePositiveRateByLabel, metrics.falsePositiveRateByLabel))

    concat_results = np.concatenate((general_score.T, label_score), axis=1)
    metrics_name = ['F-Measure', 'Precision', 'Recall', 'True_+ve_Rate', 'False_+ve_Rate']
    df = pd.DataFrame(concat_results, columns = labels, index=metrics_name)

    return acc, df


def model_testing(test_data, model):
    '''
    this function is used to get testing performance.
    metrics_train - the metrics with respect to training
    train_acc, train_score - training performance 
    metrics_test - the metrics with respect to testing
    test_acc, test_score - testing performance
    '''
    metrics_train = model.bestModel.summary
    train_acc, train_score = get_metrics(metrics_train)

    label_predictions = model.bestModel.evaluate(test_data)
    metrics_test = label_predictions
    test_acc, test_score = get_metrics(metrics_test)

    # Concatenate training and testing score
    con_df = pd.concat([train_score, test_score], axis=1, keys=[
        f'Training Accuracy = {train_acc*100:4.2f}%',
        f'Testing Accuracy = {test_acc*100:4.2f}%'
    ])

    return con_df


def draw_features_contribution(model, x_labels):
    '''
    this function is to plot feature importance.
    coeffs - the coefficients of the best model
    positive_coeffs, negative_coeffs - positive and neagative coefficients (perct)
    '''
    coeffs = model.bestModel.coefficientMatrix
    coeffs = coeffs.values

    positive_coeffs = np.array([x if x >= 0 else 0 for x in coeffs])
    negative_coeffs = np.array([x if x <= 0 else 0 for x in coeffs])

    coeffs_sum = positive_coeffs.sum() + np.absolute(negative_coeffs).sum()
    positive_coeffs /= coeffs_sum
    negative_coeffs /= coeffs_sum
    positive_coeffs *= 100
    negative_coeffs *= 100

    fig, ax = plt.subplots(figsize=(20, 9))
    ax.bar(x_labels, positive_coeffs,color='r')
    ax.bar(x_labels, negative_coeffs, color='g')
    ax.set_xlabel('Features')
    ax.set_ylabel('The user is most likely to churn (%)')
    ax.set_title('Contribution of each feature to the churn decission')
    ax.set_xticklabels(labels = x_labels, rotation='vertical')
    plt.show()


def compute_classifier_metrics(data, model, label):
    '''
    This function is used to compute classifier metrics given that the other four methods (except for logistic regression) do not have the bestModel.coefficientMatrix class.
    pred - the dataframe containing real label and its model prediction
    true_prositive, false_positive, false_negative - the score used to calculate precision, recall, and f1
    '''
    pred = model.transform(data).select('label', 'prediction')
    true_positive = pred.filter((pred.label == 1) & (pred.label == pred.prediction)).count()
    false_positive = pred.filter((pred.label == 0) & (pred.label != pred.prediction)).count()
    false_negative = pred.filter((pred.label == 1) & (pred.label != pred.prediction)).count()
    accuracy = pred.filter(pred.label == pred.prediction).count() / pred.count() * 100
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2*precision*recall / (precision + recall)
    labels = [label]
    idx = ['Accuracy', 'Precision', 'Recall', 'F-Score']
    df = pd.DataFrame(np.array([accuracy, precision, recall, f1]), columns=labels, index=idx)

    return df


def get_classifier_metrics(train, test, model):
    '''
    compute metrics of classifiers
    '''
    train_score_df = compute_classifier_metrics(train, model, 'Train')
    test_score_df = compute_classifier_metrics(test, model, 'Test')
    metrics_df = pd.concat([train_score_df, test_score_df], axis = 1)

    return metrics_df


def draw_classifier_feature_importance(model, xlabels, threshold):
    '''
    Draws a pie chart of features
    fitted_model: the fitted model
    x_labels: the labels of the features.
    threshold: the minimum value (%) to consider, 
               if the value is less than that, 
               it will be neglected (default =0)
    '''
    importance = list(model.bestModel.featureImportances.toArray())
    keep_features = [x for x in importance if x >= threshold/100]

    idx = [importance.index(x) for x in keep_features]
    features_label = [xlabels[x] for x in idx]

    # sorting
    features_label =[x for _, x in sorted(zip(keep_features, features_label))]
    keep_features = sorted(keep_features)

    # Draw
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pie(keep_features[::-1], labels=features_label[::-1] , 
           autopct='%1.1f%%', shadow=True,  
           startangle=90)
    ax.set_title('Importance of each feature to the churn decission')
    ax.axis('equal')
    plt.show()


def apply_model(train, test, feature_labels, model_type, parameter_grid=None, save_model = None, load_from_existing = None):
    '''
    This function is the machine learning model pipeline. We trained five models in this section including:
    Logistic Regression, Decision Tree Classifier, Gradient Boosting, Random Forest, and Multilayer Perceptron Classifier.
    We also calculate both training and testing metrics, and draw figures to check feature importance.
    '''
    # if a model save path is assigned 
    if save_model == None:
        save_model = f'models/{model_type}.model'

    # logistic regression
    if model_type == 'LogisticRegression':
        if load_from_existing is None:
            model = LogisticRegression()
            # define hyperparameters search window
            param_grid = ParamGridBuilder() \
                        .addGrid(model.regParam, [0.01, 0.1]) \
                        .addGrid(model.elasticNetParam, [0.0, 0.5]) \
                        .build()
            if parameter_grid is not None:
                param_grid = parameter_grid
            # train the model
            print('Logistic Regression modeling is started!')
            fit_model = model_fit(train, model, param_grid, save_model)
            print('Logistic Regression modeling is completed!')
        else:
            # load existing model
            fit_model = CrossValidatorModel.load(load_from_existing)
            
        # show metrics
        display(model_testing(test, fit_model))
        # draw feature importance
        draw_features_contribution(fit_model, feature_labels)
        
    elif model_type == 'DecisionTreeClassifier':
        if load_from_existing is None:
            model = DecisionTreeClassifier()
            # define hyperparameters search window
            param_grid = ParamGridBuilder() \
                        .addGrid(model.maxDepth, [5, 10]) \
                        .addGrid(model.impurity,['entropy', 'gini']) \
                        .build()
            if parameter_grid is not None: 
                param_grid = parameter_grid
            # train model
            print('Decision Tree training is started!')
            fit_model = model_fit(train, model, param_grid, save_model)
            print('Decision Tree training is completed!')
        else:
            # load existing model
            fit_model = CrossValidatorModel.load(load_from_existing)

        # show metrics
        display(get_classifier_metrics(train, test, fit_model))
        # draw feature importance
        draw_classifier_feature_importance(fit_model, feature_labels, threshold = 3)
        
    
    elif model_type == 'GBTClassifier':
        if load_from_existing is None:
            model = GBTClassifier()
            # define hyperparameters search window
            param_grid = ParamGridBuilder() \
                        .addGrid(model.maxDepth, [5, 10]) \
                        .addGrid(model.maxBins,[5, 10]) \
                        .addGrid(model.maxIter, [5, 10]) \
                        .build()
            if parameter_grid is not None: 
                param_grid = parameter_grid
            # train model
            print('Gradient Boosting is started!')
            fit_model = model_fit(train, model, param_grid, save_model)
            print('Gradient Boosting is completed!')
        else:
            # load model
            fit_model = CrossValidatorModel.load(load_from_existing)

        # show metrics
        display(get_classifier_metrics(train, test, fit_model))
        # draw feature importance
        draw_classifier_feature_importance(fit_model, feature_labels, threshold = 3)
    
    elif model_type == 'RandomForestClassifier':
        if load_from_existing is None:
            model = RandomForestClassifier()
            # define hyperparameters search window
            param_grid = ParamGridBuilder() \
                        .addGrid(model.maxDepth, [5, 10]) \
                        .addGrid(model.maxBins,[5, 10]) \
                        .addGrid(model.numTrees, [5, 10]) \
                        .build()
            if parameter_grid is not None: 
                param_grid = parameter_grid
            # train model
            print('Random Forest is started!')
            fit_model = model_fit(train, model, param_grid, save_model)
            print('Random Forest is completed!')
        else:
            # load model
            fit_model = CrossValidatorModel.load(load_from_existing)

        # show metrics
        display(get_classifier_metrics(train, test, fit_model))
        # draw feature importance
        draw_classifier_feature_importance(fit_model, feature_labels, threshold = 3)
    
    elif model_type == 'MultilayerPerceptronClassifier':
        if load_from_existing is None:
            model = MultilayerPerceptronClassifier()
            # define hyperparameters search window
            param_grid = ParamGridBuilder() \
                        .addGrid(model.blockSize, [64]) \
                        .addGrid(model.maxIter, [10, 20]) \
                        .addGrid(model.layers, [[98, 12, 2], [98, 5, 2]]) \
                        .addGrid(model.stepSize, [0.03]) \
                        .build()
            if parameter_grid is not None: 
                param_grid = parameter_grid
            # train model
            print('Multilayer Perceptron training is started!')
            fit_model = model_fit(train, model, param_grid, save_model)
            print('Multilayer Perceptron training is completed!')
        else:
            # load existing model
            fit_model = CrossValidatorModel.load(load_from_existing)
        # show metrics
        display(get_classifier_metrics(train, test, fit_model))
        # MultilayerPerceptronClassifier object has no attribute 'featureImportances'
        # draw_classifier_feature_importance(fit_model, feature_labels, threshold = 3)




if __name__ == "__main__":
    # the path of the raw data
    data_path = 'medium-sparkify-event-data.json'
    # save the process data to 'save_data_as'
    save_data_as = 'data/churn_model_data.csv'
    # models to be trained in this project
    model_types = ['LogisticRegression', 'DecisionTreeClassifier', 'GBTClassifier', 'RandomForestClassifier', 'MultilayerPerceptronClassifier']
    # data pipeline
    load_clean_transfer(data_path, save_data_as)
    # machine learning pipeline
    train, test, feature_labels = load_transform_ml_data(save_data_as)
    for model_type in model_types:
        apply_model(train, test, feature_labels, model_type, parameter_grid=None, save_model = None, load_from_existing = None)






