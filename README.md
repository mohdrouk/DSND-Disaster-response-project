# Disaster Response Pipeline Project
1. Installation
Python 3.6 is used to create this project and the main libraries I used are:

 - sikit-learn
 - nltk
 - Flask
 - gunicorn
 - numpy
 - pandas
 - plotly
 - sqlalchemy
 - jsonschema

2. Project Motivation
 - the project is a web application analyzing disaster data from figure 8 to build a model for an API clssifying disaster messages, 
 - the data contains messages sent during real disasters
 - the project was created through a ETL and machine learning pipelines

3. File Descriptions

 - README.md
 - ETL Pipeline Preparation.ipynb
 - ML Pipeline Preparation.ipynb
 - `app
 - run.py
 - templates
 - go.html
 - master.html
 - `data
 - DisasterResponse.db
 - disaster_categories.csv
 - disaster_messages.csv
 - process_data.py
 - `models
 - train_classifier.py
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


4. GitHub link: https://github.com/mohdrouk/DSND-Disaster-response-project