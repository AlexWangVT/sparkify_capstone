# Sparkify_Udacity data science capstone project
## Project Overview
This is the final project of the data science nanodegree (DSND). We have an example of a virtual company called 'Sparkify' who offers paid and free listening service, the customers can switch between either service, and they can cancel their subscription at any time. The given customers dataset is medium size (242M), thus the standard tools for analysis and machine learning will be useful here. For faster processing, we use Big Data tools like Apache Spark which is one of the fastest big data tools.

For this tutorial, we worked with a 242MB of the original dataset.
The problem is to create a machine learning model to predict the user intent to unsubscribe (Called customers' churn) before this happens.

## Files in this repo
* Folders
    * ***Trained models***: there are five .model folders containing each of the train models.
        - LogisticRegression.model
        - DecisionTreeClassifier.model
        - GradientBoostedTrees.model
        - RandomForestClassifier.model
        - MultilayerPerceptronClassifier.model
    * ***churn_model_data.csv***: it contains the processed data for modeling.

* Files
    * [Sparkify.ipynb](https://github.com/AlexWangVT/sparkify_capstone/blob/master/Sparkify.ipynb): The main coding file in jypyter notebook format to work in Udacity workspace.
    * [churn_prediction_modeling.py](https://github.com/AlexWangVT/sparkify_capstone/blob/master/churn_prediction_modeling.py): The main code to allpy the generalize the procedures to any dataset.
    * [README.md](https://github.com/AlexWangVT/sparkify_capstone/blob/master/README.md): this file
    * [.gitignore](https://github.com/AlexWangVT/sparkify_capstone/blob/master/.gitignore): GIT ignore file (files and folder to be ignored)

## Files NOT in this repo
* mini_sparkify_event_data.json: A 242MB json file contains the main data (exceeds GitHub limit)

## Generalization
Through the file [churn_prediction_modeling.py](https://github.com/AlexWangVT/sparkify_capstone/blob/master/churn_prediction_modeling.py), we can do the following:

**Import data pipeline to get raw data ready for modeling**
<span style="color:blue">some *blue* from churn_prediction_modeling import load_clean_transfer</span> 

**Assume we have new data new_data.json, write this command to process data:**
<span style="color:blue">some *blue* load_clean_transfer('new_data.json', save_as='new_dat_extraction')</span> 

This command will:
1. read the data from the given source,
2. clean the data
3. save the cleaned dataset to the new name as new_dat_extraction.CSV

**Next, import commands to transform the data and train/test models**
<span style="color:blue">some *blue* from churn_prediction_modeling import load_transform_ml_data, apply_model</span> 

1. **load the saved extracted data new_dat_extraction.CSV**
<span style="color:blue">some *blue* ml_ds = load_transform_ml_data(saved_as='new_dat_extraction.CSV')</span>

2. **apply an ML model to the data**
**Either by creating the model:**
<span style="color:blue">some *blue* apply_model(train, test, features_labels, model_name='GBT',save_as='NewGBT.model'))</span>

**Or by loading existing model:**
<span style="color:blue">some *blue* apply_model(train, test, features_labels,model_name='LR', load_from_existing='LogisticRegression.model'))</span>

**Note:**
You can also uncomment the main function in the churn_prediction_modeling.py file and run the script file using the command:
<span style="color:blue">some *blue* python churn_prediction_modeling.py)</span>

## Model details
Please refer to [This medium post](https://alexwangvt.github.io/sparkify_capstone/).