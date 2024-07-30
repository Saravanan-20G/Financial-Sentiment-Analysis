# Financial-Sentiment-Analysis
## Overview
This project aims to perform sentiment analysis on financial text data. The analysis includes data preprocessing, exploratory data analysis (EDA), and building a machine learning model to classify the sentiment of financial sentences.

## Dataset
The dataset contains financial sentences with labeled sentiments.
Each entry includes a Sentence and its corresponding Sentiment.

## Requirements
pandas

matplotlib

seaborn

wordcloud

nltk

scikit-learn

## Data Preprocessing
Load the dataset.

Clean the text by removing stop words, punctuation, and extra spaces.

Add new features such as word count, character count, and average word length.

## Exploratory Data Analysis (EDA)
Visualize the distribution of sentiments.

Plot the distribution of text lengths.

Generate word clouds for positive and negative sentiments.

## Text Vectorization

Use TF-IDF to convert text data into numerical features.

## Model Building

Split the data into training and test sets.

Train a RandomForestClassifier on the training data.

Evaluate the model using accuracy, confusion matrix, and classification report.


## Results
The script outputs accuracy, confusion matrix, and classification report for the model.

Visualizations are displayed for EDA and word clouds.

### Future Work
Experiment with other machine learning models.

Perform hyperparameter tuning for the RandomForestClassifier.

Add more advanced text preprocessing techniques.

### Acknowledgements
Dataset: [Dataset Source]
Libraries: pandas, matplotlib, seaborn, wordcloud, nltk, scikit-learn

this project lays the foundation for sentiment analysis in financial texts and can be extended to more complex and larger datasets for improved accuracy and insights. It demonstrates the importance of preprocessing, visualization, and robust model evaluation in the field of natural language processing (NLP).
