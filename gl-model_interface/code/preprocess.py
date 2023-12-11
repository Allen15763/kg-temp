import os, pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from processor import Preprocessor


def get_data(url):
    df = pd.read_parquet(url)
    return df


def predict(processed_data: pd.DataFrame, period: str, label: str, end_period=None) -> pd.Series:
    """
    preprocessor
    processed data: full parquet table, with temp_na.
    algorithm
    """
    output_map = {
        'label_nature': 'nature_encoder',
        'label_tag': 'tag_encoder',
        'label_cate': 'cate_encoder',
        'label_reporting': 'reporting_encoder',
        'label_mark': 'mark_encoder',
    }
    if label in output_map:
        output_encoder = output_map[label]
    else:
        raise KeyError('label error. Please confirm your input string.')

    col_features = ['ACCOUNT', 'PRODUCT',
           'RELATED_PARTY', 'JE_CATEGORY', 'LINE_DESC']
    col_labels = ['tag', 'nature', 'reporting', 'cate', 'mark']

    col_all = ['JOURNAL_DOC_NO'] + col_features + col_labels + ['period']

    preprocessor = Preprocessor()
    # Get relevant data with specific period and columns.
    if end_period == None:
        my_df_new = preprocessor.get_approximate_data(processed_data, col_all, period)
    else:
        my_df_new = preprocessor.get_approximate_data(processed_data, col_all, period, end_period)

    # Load encoders.
    preprocessor.load_encoder_set(r'C:\SEA\EAEL\programing\general_ledger\feature_encoder')
    with open(f'C:\SEA\EAEL\programing\general_ledger\label_encoder\{output_encoder}.pickle', 'rb') as f:
        label_convertor_temp = pickle.loads(pickle.load(f))

    # Transform data to tensor.
    my_df_new = preprocessor.preprocess_non_desc(my_df_new, fit=False)
    # Log missing items.
    # preprocessor.potential_ignored(my_df_new.ACCOUNT, my_df_new.cleaned_desc)
    my_df_new = preprocessor.preprocess_desc(my_df_new)


    # To separate X, Y then transforming Y. Y is temp_na in this case.
    train_data_features, train_data_targets = preprocessor.get_processed_data(my_df_new, label=label)
    # test_x, _ = train_data_features, train_data_targets
    del my_df_new
    tr_df = train_data_features.assign(target=train_data_targets)
    # tr_df.to_csv("../input/train.csv", index=False)
    tr_df.to_csv("train.csv", index=False)



if __name__ == '__main__':
    import time
    import warnings
    warnings.simplefilter("ignore")
    a = time.time()
    path = r'\\DS01\1_Department\FN 財務部\Shopee Project\Allen\EAEL\reversion\output tagged summary\SPT_202308_withADJ_20230914.parquet'
    target_label = 'label_nature'
    period = '2023-08'
    end_period = None

    df = get_data(path)
    # For 2018 unencoded labels.
    df['PRODUCT'] = np.where((df.PRODUCT == 'EC_COM')|\
                             (df.PRODUCT == 'EC_SPE')|\
                             (df.PRODUCT == 'RT_B2C')
                             , '000', df.PRODUCT)
    df['PRODUCT'] = np.where((df.PRODUCT == 'LG_SPX'), 'LG_SPX_OWN', df.PRODUCT)

    df['JE_CATEGORY'] = np.where((df.JE_CATEGORY == 'Reclass'), 'Transfer', df.JE_CATEGORY)

    df['RELATED_PARTY'] = np.where((df.RELATED_PARTY == 'GTW'), 'MOBTW', df.RELATED_PARTY)
    df['RELATED_PARTY'] = np.where((df.RELATED_PARTY == 'UTLSG'), 'UTLUS', df.RELATED_PARTY)
    df['RELATED_PARTY'] = np.where((df.RELATED_PARTY == 'SPL')|(df.RELATED_PARTY == 'GIH'), '000', df.RELATED_PARTY)
    # For 202306 New Product code on Welfare Employee Insurance and New Party, ignore
    df['PRODUCT'] = np.where((df.PRODUCT == 'EC_SI_IMH'), '000', df.PRODUCT)
    df['RELATED_PARTY'] = np.where((df.RELATED_PARTY == 'SHPLP'), '000', df.RELATED_PARTY)

    # 202307 unexpected PRODUCT code.(202308:BE_LEG) Remove irrelevant record to avoid from memory explosion.
    df['PRODUCT'] = np.where((df.PRODUCT.isin(['BE_COO', 'BE_CORPIT', 'BE_LEG']) ), '000', df.PRODUCT)
    # df = df.drop(df.loc[df.CREATED_BY == 'YEHLY', :].index).reset_index()
    predict(df, period, target_label, end_period)


    print(f'done. {time.time()-a} s') # 306.46 s


