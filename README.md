# Disaster Response Pipeline

## Table of Contents

- [Overview](#overview)
- [Application components](#components)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Web App](#flask)



***

<a id='overview'></a>

## Overview

This application, is intended to analyze disaster messages from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> and to build a model for predict categories of a disaster message.

_workspace_ directory consists of a few folder with scripts and data.


<a id='components'></a>

## Application Components

The application includes three components:

<a id='etl_pipeline'></a>

### ETL Pipeline

A Python script in _workspace/data/process_data.py_ contains some functions:

- to load data from csv files
- to create pandas dataframes and clean data
- to merge dataframes and save data into a table in a sql database

<a id='ml_pipeline'></a>

### ML Pipeline

A Python script in _workspace/models/train_classifier.py_ contains some functions:

- to load data from a table on a sql database
- to tokenize and lemminize a text data
- to build a model to predict a category message
- to predict a test data
- to save model into a pickle file

<a id='flask'></a>

### Web App

To run the web application you have to complete three command:
* `python workspace/data/process_data.py workspace/data/disaster_messages.csv workspace/data/disaster_categories.csv workspace/data/DisasterResponse.db`

    The first argument is a path to the script file, second and third are csv files and the fourth is a path to save data.
* `python workspace/models/train_classifier.py workspace/data/DisasterResponse.db workspace/models/classifier.pkl`
    
    The first argument is a path to the script file, the second is a path to a sql database and the third is a path to save model to a pickle file.

* `python run.py`
    
    This command will start the web app.
