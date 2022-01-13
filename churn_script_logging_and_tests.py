'''
A module to test predict customer churn

Author: Pemberai Sweto
Date: January, 2022
'''

import os
import logging
import joblib
import pytest
import churn_library

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name='df_initial')
def df_initial():
    """
    initial dataframe fixture
    """
    try:
        df_initial_result = churn_library.import_data("./data/bank_data.csv")
        logging.info("Initial dataframe fixture creation: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Initial fixture creation: The file wasn't found")
        raise err
    return df_initial_result


@pytest.fixture(name='df_encoded')
def df_encoded(df_initial_data):
    """
    encoded dataframe fixture
    """
    try:
        df_encoded_result = churn_library.encoder_helper(
            df_initial_data, ["Gender",
                         "Education_Level",
                         "Marital_Status",
                         "Income_Category",
                         "Card_Category"])
        logging.info("Encoded dataframe fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoded dataframe fixture creation: It was not possible to encode some columns")
        raise err
    return df_encoded_result


@pytest.fixture(name='feature_engineering')
def feature_engineering(df_encoded_data):
    """
    feature_engineering fixtures
    """
    try:
        x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(
            df_encoded_data)
        logging.info("Feature engineering fixture creation: SUCCESS")
    except BaseException as err:
        logging.error(
            "Feature engineering fixture creation: Features lengths mismatch")
        raise err
    return x_train, x_test, y_train, y_test


def test_import(df_initial_data):
    '''
    test data import
    '''
    try:
        assert df_initial_data.shape[0] > 0
        assert df_initial_data.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_initial_data):
    '''
    test perform eda function
    '''
    churn_library.perform_eda(df_initial_data)
    images_list = [
        "churn_hist",
        "customer_age_hist",
        "marital_status_bar",
        "total_trans_ct_dist",
        "correlation_heatmap"]
    for image_name in images_list:
        try:
            with open(f'./images/eda/{image_name}.png', 'r', encoding='utf8'):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: Some of the images is missing")
            raise err


def test_encoder_helper(df_encoded_data):
    '''
    test encoder helper
    '''
    try:
        assert df_encoded_data.shape[0] > 0
        assert df_encoded_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have rows and columns")
        raise err

    try:
        encoded_columns = ["Gender",
                           "Education_Level",
                           "Marital_Status",
                           "Income_Category",
                           "Card_Category"]
        for column in encoded_columns:
            assert column in df_encoded_data
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")
    return df_encoded_data


def test_perform_feature_engineering(feature_engineering_data):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train = feature_engineering_data[0]
        x_test = feature_engineering_data[1]
        y_train = feature_engineering_data[2]
        y_test = feature_engineering_data[3]
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Features lengths mismatch")
        raise err
    return feature_engineering_data


def test_train_models(feature_engineering_data):
    '''
    test train_models
    '''
    churn_library.train_models(
        feature_engineering_data[0],
        feature_engineering_data[1],
        feature_engineering_data[2],
        feature_engineering_data[3])
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The file was not found")
        raise err
    images_list = ["rfc_report_test",
                   "rfc_report_train",
                   "lrc_report_test",
                   "lrc_report_train",
                   "feature_importance"]
    for image_name in images_list:
        try:
            with open(f'./images/results/{image_name}.png', 'r', encoding='utf8'):
                logging.info("Testing train_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models:  Some of the images is missing")
            raise err


if __name__ == "__main__":
    pass
