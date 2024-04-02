# Title: Intent-based Chatbot with Random Forest Classifier

# Introduction:
This Python script implements a simple chatbot using a Random Forest Classifier trained on intent data. The chatbot provides responses based on the user's input, utilizing a bag-of-words model and a classification algorithm. The Flask web framework is employed to create a user interface for interacting with the chatbot.

# Code Overview:
- Importing Libraries:
  NumPy for numerical computations.
  Matplotlib for plotting graphs.
  Pandas for data manipulation.
  NLTK for natural language processing tasks.
  Random for generating random responses.
  Flask for web application development.
- Loading Intent Dataset:
  Intent data is loaded from the 'intend.json' file using Pandas.
  Data is preprocessed by tokenization, stemming, and removing stopwords.
- Bag-of-Words Model:
  CountVectorizer from sklearn.feature_extraction.text is used to convert text data into a numerical format.
- Train-Test Split:
  The dataset is split into training and testing sets using train_test_split from sklearn.model_selection.
- Random Forest Classifier:
  RandomForestClassifier from sklearn.ensemble is applied to train the classification model.
- Model Evaluation:
  Accuracy of the model is evaluated using confusion_matrix and accuracy_score from sklearn.metrics.
- Chatbot Function:
  User input is processed to extract features using the trained CountVectorizer.
  The classifier predicts the intent based on the input.
  Random responses corresponding to the predicted intent are selected and returned.
- Flask Web Application:
  Flask routes are defined to handle user requests and display chatbot responses.
  A simple HTML interface is rendered using render_template.
# Conclusion:
This script demonstrates the implementation of a basic intent-based chatbot using a Random Forest Classifier. The chatbot can understand user queries and provide relevant responses based on predefined intents. Integration with a web interface via Flask allows for easy interaction with the chatbot.

#  Dependencies:
- NumPy
- Matplotlib
- Pandas
- NLTK
- Flask

# Instructions:
- Ensure the 'intend.json' file containing intent data is in the specified directory.
- Install the required dependencies if not already installed.
- Run the script to start the Flask web application.
- Access the chatbot interface via a web browser and interact by entering text queries.
  
# Author:
Prajesh Tejani

# References:
- NLTK documentation: https://www.nltk.org/
- Flask documentation: https://flask.palletsprojects.com/


