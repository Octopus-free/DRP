import sys
# import libraries
import pandas as pd
import re
from sqlalchemy import create_engine
import sqlite3
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def load_data(database_filepath):
    """
        Load data from a SQL Database
        Args:
        database_filepath: path to a sql database file
        Returns:
        X array: a array with messages
        Y array: a array with labels
        category_names: a list with label names
    """

    # load data from database into a datafrmae
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)

    # create list with label names
    category_names = df.columns[5:]

    # convert dataframes to arrays
    X = df[['message']].values[:, 0]
    Y = df[category_names].values

    return X, Y, category_names


def tokenize(text):

    """
        Tokenize text data
        Args:
        text str: a single message to tokenize
        Returns:
        tokens list: a tokenized text
    """

    # detecting URL
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
        Build model to predict

        Returns:
        Trained model on messages
    """

    # model pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ]))

        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                             n_jobs=-1, random_state=42)))
    ])
    
    parameters = {
    'features__text_pipeline__vect__max_features': (None, 5000, 10000),
    'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'features__text_pipeline__tfidf__use_idf': (True, False)
              }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
        Predict test data and generate a report
        Args:
        model: a trained model
        X_test: test meaasages
        Y_test: test labels
        category_names: label names to report
    """

    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):

    """
        Save a model to a pickle file
        Args:
        model: a trained model
        model_filepath: a path to save the model
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
