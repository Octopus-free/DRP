import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """
        - Two csv files as input data
        - Create two separate pandas dataframes based on data from each csv file.
        - Merge two dataframes into a final dataframe
        Args:
        messages_file_path str: csv file with messages
        categories_file_path str: csv file with category labels
        Returns:
        dataframe: final dataframe with all meassages and labels
    """
    # read csv with messages
    messages = pd.read_csv(messages_filepath)
    # read csv with category labels
    categories = pd.read_csv(categories_filepath)
    # merge two dataframe into a final dataframe
    df = messages.merge(categories, left_on='id', right_on='id', how='outer')

    return df


def clean_data(df):
    """
        - Clean an input dataframe

        Args:
        df: a final dataframe created by load_data() function
        Returns:
        df: a cleaned dataframe
    """
    # split one category column to a set of column (one column to each label)
    categories = df['categories'].str.split(';', expand=True)
    # take a first row to create columns names
    row = categories.head(1)
    # create column names
    category_colnames = []
    for x in range(row.shape[1]):
        category_colnames.append(row[x][0][:-2])

    # rename column names in categories dataframe
    categories.columns = category_colnames

    # replace data, leave only last character in data
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    
    # drop row with 'related'=2
    categories.drop(categories[categories['related'] == 2].index, inplace=True)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # merge the massage dataframe with category dataframe
    df = df.merge(categories, left_index=True, right_index=True)
    # drop duplicates from a final dataframe
    df.drop_duplicates('id', inplace=True)

    return df


def save_data(df, database_filename):
    """
        Save a cleaned dataframe to a SQL database
        Args:
        df: cleaned dataframe created by clean_data() function
        database_file_name: path to a SQL Database to save data into a table
        Returns:
        None
    """
    # create connection to a sql database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save a dataframe to table messages in a sql database
    df.to_sql('messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
