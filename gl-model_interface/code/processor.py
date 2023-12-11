import pandas as pd
import numpy as np
import os, re
import io
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle


class Preprocessor:
    def __init__(self) -> None:
        self.encoders_name = ['product_encoder', 'party_encoder', 'je_category_encoder', 'line_desc_encoder'
            , 'account_encoder']

    def get_approximate_data(self, df, cols: list, beg_period: str, end_period=None) -> pd.DataFrame:
        if end_period == None:
            end_period = beg_period
        else:
            pass
        df = df.loc[pd.to_datetime(df.period).between(datetime.strptime(beg_period, '%Y-%m')
                                                      , datetime.strptime(end_period, '%Y-%m'))
                    , cols].reset_index(drop=True)
        return df

    @classmethod
    def clean_desc(cls, data_series):
        # pattern1 = '\D|(7\-11)|(711)'
        pattern1 = '\D'  # RECON_Fubon #9251、EA_富邦9251、EA_CTBC 4935、RECON_Local-3PL-給的運費返還，will be effected.
        p = re.compile(pattern1)
        pattern2 = '[^\/\.\、\(\)\&\[\]]'
        p2 = re.compile(pattern2)
        data_series = data_series.apply(lambda x: re.sub('7-11', 'seven', x))
        data_series = data_series.apply(lambda x: re.sub('711', 'seven', x))
        data_series = data_series.apply(lambda x: re.sub('美聯社', '美廉社', x))
        data_series = data_series.apply(lambda x: re.sub('(?i)Cross Border|Cross-Border', 'CB', x))

        new_series = data_series.apply(lambda x: ''.join(p.findall(x)))
        new_series = new_series.apply(lambda x: ''.join(p2.findall(x)))
        new_series = new_series.apply(lambda x: x.strip() if re.match('^ ', str(x)) != None else x)
        new_series = new_series.apply(lambda x: x.strip()[1:] if re.match('^-|^_', str(x)) != None else x.strip())
        new_series = new_series.apply(lambda x: x.strip() if re.match('^ ', str(x)) != None else x)
        new_series = new_series.apply(lambda x: x.strip()[1:] if re.match('^-|^_', str(x)) != None else x.strip())
        new_series = new_series.apply(lambda x: x.strip()[1:] if re.match('^-|^_', str(x)) != None else x.strip())

        # 取代中間或後面沒抓到的特殊字元與多餘空白
        new_series = new_series.apply(lambda x: re.sub('_|-|,|:', ' ', x))
        new_series = new_series.apply(lambda x: re.sub(r'\s{2,}', ' ', x)) \
            .apply(lambda x: x.strip())
        new_series = new_series.apply(lambda x: re.sub('\u2765', '', x))

        # Remove description from other project  e.g. _[content].；exclude at beginning 20230302
        def cleaning_label_from_others(string):
            try:
                result = re.search('[^\[\]]+', string).group()
                return result
            except:
                return string

        new_series = new_series.apply(cleaning_label_from_others)
        new_series = new_series.apply(lambda x: re.sub('_$', '', x))

        return new_series

    def preprocess_non_desc(self, df, fit=False):
        """
        after processed, features left,
        'PRODUCT', 'RELATED_PARTY', 'JE_CATEGORY', 'LINE_DESC', 'tag', 'nature', 'reporting', 'cate', 'mark'
        , 'acc_period', '100031'........'Manual Adjustment', 'No data', 'has_datetime', 'cleaned_desc'
        """

        def catch_datetime_series(series):
            """
            yyyy/mm/dd
            yyyy/mm
            yyyy-mm-dd
            yyyy/mm-yyyy/mm
            yyyy/mm/dd-yyyy/mm/dd
            """
            pattern = r"(\d{4})([-/])(\d{2})(?:\2(\d{2}))?(?:[-/](\d{2}))?(?:[-/](\d{4}))?(?:[/](\d{2}))?(?:[/](\d{4}))?(?:[-/](\d{2}))?(?:[-/](\d{4}))?"

            def datetime_format(string):
                try:
                    result = re.search(pattern, string).group()
                    return 1
                except:
                    return 0

            new_series = series.apply(datetime_format)
            return new_series

        # df = df.join(pd.get_dummies(df.ACCOUNT, prefix='ACCOUNT')).drop('ACCOUNT', axis=1)

        if fit:
            df = df.assign(
                PRODUCT=self.product_encoder.fit_transform(df.PRODUCT),
                RELATED_PARTY=self.party_encoder.fit_transform(df.RELATED_PARTY),
                JE_CATEGORY=self.je_category_encoder.fit_transform(df.JE_CATEGORY),
                has_datetime=catch_datetime_series(df.LINE_DESC),
                cleaned_desc=self.clean_desc(df.LINE_DESC)
            )
            df.rename(columns={
                'period': 'acc_period'
            }, inplace=True)
        else:
            df = df.assign(
                PRODUCT=self.product_encoder.transform(df.PRODUCT),
                RELATED_PARTY=self.party_encoder.transform(df.RELATED_PARTY),
                JE_CATEGORY=self.je_category_encoder.transform(df.JE_CATEGORY),
                has_datetime=catch_datetime_series(df.LINE_DESC),
                cleaned_desc=self.clean_desc(df.LINE_DESC)
            )

        if 'tag' in df.columns:
            df.rename(columns={
                'period': 'acc_period',
                'tag': 'label_tag',
                'nature': 'label_nature',
                'reporting': 'label_reporting',
                'cate': 'label_cate',
                'mark': 'label_mark',
            }, inplace=True)
        df.pop('LINE_DESC')
        return df

    def build_encoder_set(self, *args):
        if len(args) != 0:
            self.product_encoder = args[0]
            self.party_encoder = args[1]
            self.je_category_encoder = args[2]
            self.line_desc_encoder = args[3]
            self.account_encoder = args[4]
        else:
            self.product_encoder = LabelEncoder()
            self.party_encoder = LabelEncoder()
            self.je_category_encoder = LabelEncoder()
            self.line_desc_encoder = CountVectorizer()
            self.account_encoder = OneHotEncoder(handle_unknown='ignore')

    def load_encoder_set(self, path=None):
        if path == None:
            file_product = open(os.path.join('', self.encoders_name[0]+ '.pickle'), 'rb')
            self.product_encoder = pickle.load(file_product)
            file_product.close()

            file_party = open(os.path.join('', self.encoders_name[1]+ '.pickle'), 'rb')
            self.party_encoder = pickle.load(file_party)
            file_party.close()

            file_je_category = open(os.path.join('', self.encoders_name[2]+ '.pickle'), 'rb')
            self.je_category_encoder = pickle.load(file_je_category)
            file_je_category.close()

            file_line_desc = open(os.path.join('', self.encoders_name[3]+ '.pickle'), 'rb')
            self.line_desc_encoder = pickle.load(file_line_desc)
            file_line_desc.close()

            file_acc = open(os.path.join('', self.encoders_name[4]+ '.pickle'), 'rb')
            self.account_encoder = pickle.load(file_acc)
            file_acc.close()

        else:
            file_product = open(os.path.join(path, self.encoders_name[0]+ '.pickle'), 'rb')
            self.product_encoder = pickle.load(file_product)
            file_product.close()

            file_party = open(os.path.join(path, self.encoders_name[1]+ '.pickle'), 'rb')
            self.party_encoder = pickle.load(file_party)
            file_party.close()

            file_je_category = open(os.path.join(path, self.encoders_name[2]+ '.pickle'), 'rb')
            self.je_category_encoder = pickle.load(file_je_category)
            file_je_category.close()

            file_line_desc = open(os.path.join(path, self.encoders_name[3]+ '.pickle'), 'rb')
            self.line_desc_encoder = pickle.load(file_line_desc)
            file_line_desc.close()

            file_acc = open(os.path.join(path, self.encoders_name[4]+ '.pickle'), 'rb')
            self.account_encoder = pickle.load(file_acc)
            file_acc.close()

    def load_encoder_set_s3(self, s3_client):
        ob = s3_client.get_object(Bucket="sg-seafin-tw-notebook",
                                  Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/feature_encoder/{self.encoders_name[0]}.pickle")
        self.product_encoder = pickle.loads(ob['Body'].read())

        ob = s3_client.get_object(Bucket="sg-seafin-tw-notebook",
                                  Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/feature_encoder/{self.encoders_name[1]}.pickle")
        self.party_encoder = pickle.loads(ob['Body'].read())

        ob = s3_client.get_object(Bucket="sg-seafin-tw-notebook",
                                  Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/feature_encoder/{self.encoders_name[2]}.pickle")
        self.je_category_encoder = pickle.loads(ob['Body'].read())

        ob = s3_client.get_object(Bucket="sg-seafin-tw-notebook",
                                  Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/feature_encoder/{self.encoders_name[3]}.pickle")
        self.line_desc_encoder = pickle.loads(ob['Body'].read())

        ob = s3_client.get_object(Bucket="sg-seafin-tw-notebook",
                                  Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/feature_encoder/{self.encoders_name[4]}.pickle")
        self.account_encoder = pickle.loads(ob['Body'].read())

    def load_s3(self, s3_client, *args):
        ob = s3_client.get_object(Bucket="sg-seafin-tw-notebook", Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/{args[0]}")
        data = pickle.loads(ob['Body'].read())
        return data

    def preprocess_desc(self, df, fit=False):
        data = df.pop('cleaned_desc')
        data_acc = df.pop('ACCOUNT')
        if fit:
            parse_matrix_desc = self.line_desc_encoder.fit_transform(data)
            parse_matrix_acc = self.account_encoder.fit_transform(data_acc.values.reshape(-1, 1))
        else:
            parse_matrix_desc = self.line_desc_encoder.transform(data)
            parse_matrix_acc = self.account_encoder.transform(data_acc.values.reshape(-1, 1))

        desc_encoded = pd.DataFrame(data=parse_matrix_desc.todense()
                                    , columns=self.line_desc_encoder.get_feature_names_out())
        acc_encoded = pd.DataFrame(data=parse_matrix_acc.todense()
                                   , columns=self.account_encoder.categories_[0])
        df_processed = df.join(acc_encoded)
        df_processed = df_processed.join(desc_encoded)
        return df_processed

    def save_encoders(self, path=None):
        """
        Saved the main py file where is, if there is no path.
        """
        encoders = [self.product_encoder, self.party_encoder, self.je_category_encoder, self.line_desc_encoder
            , self.account_encoder]
        if path != None:
            for name, encoder in zip(self.encoders_name, encoders):
                with open(os.path.join(path, name), 'wb') as f:
                    pickle.dump(encoder, f)
        else:
            for name, encoder in zip(self.encoders_name, encoders):
                with open(os.path.join(name), 'wb') as f:
                    pickle.dump(encoder, f)

    def save_encoders_s3(self, s3_client):
        """
        Using s3 SDK object to storage serialized data.
        """
        encoders = [self.product_encoder, self.party_encoder, self.je_category_encoder, self.line_desc_encoder
            , self.account_encoder]
        for name, encoder in zip(self.encoders_name, encoders):
            ob = pickle.dumps(encoder)

            response = s3_client.put_object(
                Bucket="sg-seafin-tw-notebook", Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/{name}.pickle", Body=ob
            )

            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            if status == 200:
                print(f"Successful S3 put_object response. Status - {status}, object name: {name}")
            else:
                print(f"Unsuccessful S3 put_object response. Status - {status}, object name: {name}")

    def save_s3(self, s3_client, *args):
        """
        Using s3 SDK object to storage serialized data.
        """
        ob = pickle.dumps(args[1])

        response = s3_client.put_object(
            Bucket="sg-seafin-tw-notebook", Key=f"workspaces/{os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]}/src/{args[0]}.pickle", Body=ob
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}, object name: {args[0]}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}, object name: {args[0]}")

    def save_processed_data(self, df, path=None):
        df.to_parquet(f'processed_data_all_{int(datetime.today().timestamp())}.parquet')

    def inverse(self, **kwargs):
        pass

    def potential_ignored(self, se, se2):
        ignored_acc = [i for i in se.unique() if i not in self.account_encoder.categories_[0]]
        with open(f'ignored_acc{int(datetime.today().timestamp())}.txt', 'w', encoding='utf-8') as f:
            for i in ignored_acc:
                f.write(f'{i}\n')
        print(f'ignored ACCOUNT: {ignored_acc}')

        ignored_desc = [i for i in set(' '.join([str(i) for i in se2.unique()]).split(' ')) \
                        if i not in self.line_desc_encoder.get_feature_names_out()]
        with open(f'ignored_desc_words{int(datetime.today().timestamp())}.txt', 'w', encoding='utf-8') as f:
            for i in ignored_desc:
                f.write(f'{i}\n')
        print(f'ignored desc words num: {len(ignored_desc)}')

    def potential_ignored_s3(self, se, se2, s3_client):
        ignored_acc = [i for i in se.unique() if i not in self.account_encoder.categories_[0]]
        ignored_desc = [i for i in set(' '.join([str(i) for i in se2.unique()]).split(' ')) \
                        if i not in self.line_desc_encoder.get_feature_names_out()]
        print(f'ignored ACCOUNT: {ignored_acc}')
        print(f'ignored desc words num: {len(ignored_desc)}')

        df_acc = pd.Series(ignored_acc, name='ignored_acc') # Accounts weren't covered by Encoder.(Onehot)
        df_desc = pd.Series(ignored_desc, name='ignored_desc')  # clean_desc weren't covered by Encoder.(CountVector)

        notebook_loc = os.environ['NOTEBOOK_ALLUXIO_WORKSPACE_URI'].split('/')[-1]
        folder = datetime.strftime(datetime.today(), '%Y-%m-%d')

        data_set = [df_acc, df_desc]
        for i, d in enumerate(data_set):
            if i == 0:
                file_name = f'ignored_acc_{int(datetime.today().timestamp())}'
            elif i == 1:
                file_name = f'ignored_desc_words_{int(datetime.today().timestamp())}'

            with io.StringIO() as csv_buffer:
                d.to_csv(csv_buffer, index=False)

                response = s3_client.put_object(
                    Bucket="sg-seafin-tw-notebook"
                    , Key=f"workspaces/{notebook_loc}/dataset/{folder}/{file_name}.csv",
                    Body=csv_buffer.getvalue()
                )

            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 200:
                print(f"Successful S3 put_object response. Status - {status}, file: {file_name}")
            else:
                print(f"Unsuccessful S3 put_object response. Status - {status}, file: {file_name}")

    def get_processed_data(self, df, label='label_nature'):
        target = df.loc[:, label]
        df = df.drop(['JOURNAL_DOC_NO', 'label_tag', 'label_nature', 'label_reporting', 'label_cate'
                         , 'label_mark', 'acc_period'], axis=1)
        return df, target


class Mapper:
    def __init__(self, df_raw: pd.DataFrame, url_modified_file: str, period: str, df_manual_adj, sample_number=25000):
        self.df_raw = df_raw
        self.url_modified_file = url_modified_file
        self.period = period
        self.sample_number = sample_number
        self.df_manual_adj = df_manual_adj

    def get_manual_modified(self) -> pd.DataFrame:
        """
        Get modified file from Excel GL. e.g. SPTTW GL-202306_manua.xlsx
        The modified excel sheet should be named GL.
        :return: A GL added tag, nature, and other labeling columns.
        """
        df = pd.read_excel(self.url_modified_file, sheet_name='GL', dtype=str)
        df = df.assign(
            ACCOUNTED_DR_SUM = df['Accounted Dr SUM'].astype('int64'),
            ACCOUNTED_CR_SUM = df['Accounted Cr SUM'].astype('int64'),
            LINE_DESC = df['LINE_DESC'].fillna('No data')
        )
        self.df_new = df
        return self.df_new

    def checking_samples(self) -> list:
        """
        Note: Do not use df with manual adjustment.

        To compare labeled excel and parquet(warehouse) to ensure casting process is correct.

        :param df_new: extracted from modified excel.
        :param df_raw: data warehouse without concat manual adjustment.
        :param period: GL month.
        :param sample_number:
        :return: OPT. list
        """
        df_raw = self.df_raw.loc[self.df_raw.period == self.period, :].reset_index(drop=True).copy()
        assert df_raw.shape[0] == self.df_new.shape[0], 'ISSUE, length is different.'

        sample = df_raw.sample(self.sample_number).index.values
        result = []
        for i in sample:
            a = self.df_new.iloc[i, [*range(3, 5)
                , self.df_new.columns.get_loc('JE_CATEGORY')
                , self.df_new.columns.get_loc('JOURNAL_DOC_NO')
                , self.df_new.columns.get_loc('LINE_DESC')
                , self.df_new.columns.get_loc('ACCOUNTED_DR_SUM')
                , self.df_new.columns.get_loc('ACCOUNTED_CR_SUM')]
            ]
            b = df_raw.iloc[i, [df_raw.columns.get_loc('ACCOUNT')
                , df_raw.columns.get_loc('ACCOUNT_DESC')
                , df_raw.columns.get_loc('JE_CATEGORY')
                , df_raw.columns.get_loc('JOURNAL_DOC_NO')
                , df_raw.columns.get_loc('LINE_DESC')
                , df_raw.columns.get_loc('ACCOUNTED_DR_SUM')
                , df_raw.columns.get_loc('ACCOUNTED_CR_SUM')]
            ]
            result.append(tuple([all(a.values == b.values), i]))
        print(f'Sampled: {len(sample)}\nIssue cases: {len([i for i in result if i[0] != True])}')
        return result

    def get_labels(self) -> pd.DataFrame:
        """
        Casting labels from new to data warehouse table After check completed and no Error.
        :param df_new: checked GL.
        :param df_raw: data warehouse.
        :param period: GL month.
        :return: full GL dataframe.
        """
        df_new = self.df_new.copy()
        df_raw = self.df_raw.copy()

        sliced_df = df_raw.loc[df_raw.period == self.period, :].reset_index(drop=True).copy()
        sliced_df = sliced_df.assign(
            tag=df_new.tag,
            nature=df_new.nature,
            reporting=df_new.reporting,
            cate=df_new.cate,
            mark=df_new.mark
        )
        df = pd.concat([df_raw.loc[df_raw.period != self.period, :], sliced_df], ignore_index=True)
        return df

    @staticmethod
    def export_manuan_adjustment_data_for_modiying(url_get: str, url_export: str):
        # Read parquet, to_excel. Next, get_renewed_manuan_adjustment_data
        df = pd.read_parquet(url_get)
        df.to_excel(url_export, index=False)

    @staticmethod
    def get_renewed_manuan_adjustment_data(url_get: str, url_export: str) -> pd.DataFrame:
        df = pd.read_excel(url_get, dtype=str)
        df['amt'] = df.amt.astype('int64')
        df['ACCOUNTED_DR_SUM'] = df.ACCOUNTED_DR_SUM.astype('int64')
        df['ACCOUNTED_CR_SUM'] = df.ACCOUNTED_CR_SUM.astype('int64')
        df.to_parquet(url_export)

    def run(self):
        self.get_manual_modified()
        history = self.checking_samples()
        df_mapped = self.get_labels()
        df_mapped = pd.concat([df_mapped, self.df_manual_adj], ignore_index=True)
        return df_mapped


class Pivoter:

    @staticmethod
    def pivot_details(df):
        # 133 cates. 20230515. 20230712 added #0326 relevant accounts.
        result = df.query("tag.isin(['EA', 'EL', 'RECON'])").pivot_table(
            values='amt',
            index=['tag', 'cate', 'nature'],
            columns='period',
            aggfunc=np.sum
        ).fillna(0)
        result['total'] = result.apply(lambda x: sum(x), axis=1)
        result = result.applymap(lambda x: format(int(x), ',')).reset_index()

        # ReOrder
        ea = result.iloc[:15]
        el = result.iloc[15:35]
        recon = result.iloc[35:]

        conditions = [(ea['nature'] == '112001-Escrow Bank held by SPETW for SPTTW'),
                      (ea['nature'] == '112001-Escrow offline-新光#0326收款'),
                      (ea['nature'] == '112001/112002-SPS Bank for CB transactions'),
                      (ea['nature'] == '101851/101911-SPTTW Escrow Bank for SPX COD & Kmart consignment goods'),
                      (ea['nature'] == '101824-SPTTW Escrow Bank for offline adjustments'),
                      (ea['nature'] == '113101-Receivables from payment gateway'),
                      (ea['nature'] == '113101-Receivables from payment gateway-CB Escrow儲值from SVS'),
                      (ea['nature'] == '113103/112002-Cash in Payment Gateway'),
                      (ea['nature'] == '110003-Receivables from COD'),
                      (ea['nature'] == '112501-Receivables from JKO'),
                      (ea['nature'] == '100118-COD Cash in Service Points'),
                      (ea['nature'] == '111421-E-commerce Manual Settlement Receivables - Others'),
                      (ea['nature'] == '111101-SPX COD in-transit in Parnership'),
                      (ea['nature'] == 'SCM collection'),
                      (ea['nature'] == 'SCM consignment offline refuund')]
        then = [*range(15)]

        ea['order'] = np.select(conditions, then)
        ea = ea.sort_values(by=['order']).drop('order', axis=1)

        # sort SPS
        sps = recon.query("cate=='WC in SPS Bank'")
        sps = pd.concat([
            sps.query("nature=='Revenue-MP'"),
            sps.query("nature=='CB PRM'"),
            sps.query("nature=='111406-Logs Prepayment-CB'"),
            sps.query("nature=='112002-SPS尚未移轉'"),
            sps.query("nature=='200409-Custom taxes and Fraud cases'"),
            sps.query("nature=='200409-Custom taxes and Fraud cases_write-off'"),
            sps.query("nature=='CB 關稅代收付'"),
            sps.query("nature=='200701-Accrued Logs Payable (SLS)'"),
            sps.query("nature=='200701-Accrued Logs Payable (SLS)-Payments'"),
            # sps.query("nature=='CB Escrow儲值from SVS'"),
            sps.query("nature=='CB 撥款'"),
            sps.query("nature=='SPS代收付'"),
            sps.query("nature=='SPS代收付-撥款'"),
            sps.query("nature=='CB 運費'"),
            sps.query("nature=='CB 運費返還'"),
            sps.query("nature=='CB SFR'"),
            sps.query("nature=='CB SVS 收款-PKG/ Paid ads/coin'"),
            sps.query("nature=='CB offline adjustment'"),
            sps.query("nature=='Local SIP'"),
            sps.query("nature=='CB匯費'"),
            sps.query("nature=='CB 手續費'"),
            sps.query("nature=='委託SPS'"),
            sps.query("nature=='補SPS'"),
            sps.query("nature=='CB LM 運費'"),
            sps.query("nature=='CB Escrow儲值from SVS - Reclassify'")
        ], ignore_index=True)

        # sort recon
        recon = pd.concat([recon.query("cate=='WC in Escrow Bank - Withdrawal'"),
                           recon.query("cate=='WC in Escrow Bank - Top up to escrow bank'"),
                           recon.query("cate=='WC in Escrow Bank - Offline'"),
                           sps,
                           recon.query("cate=='WC in JKO company'"),
                           recon.query("cate=='WC in SPESZ Bank'"),
                           recon.query("cate=='Recon - SPX'"),
                           recon.query("cate=='Recon - Others'"),
                           recon.query("cate.isin(['Recon - Others_N', 'Recon - 8633', 'Recon - 8850', 'Recon - 9641'])"),
                           recon.query("cate=='Manual Adjustment'")], ignore_index=True)
        result = pd.concat([ea, el, recon], ignore_index=True)
        return result

    @staticmethod
    def pivot_reporting(df):
        result = df.query("tag.isin(['EA', 'EL', 'RECON'])").pivot_table(
            values='amt',
            index=['tag', 'reporting'],
            columns='period',
            aggfunc=np.sum
        ).fillna(0)
        result['total'] = result.apply(lambda x: sum(x), axis=1)
        result = result.applymap(lambda x: format(int(x), ',')).reset_index()
        return result

    @staticmethod
    def pivot_rev(df):
        recon_details = ['CC_Promo CB', 'CC_Promo Local', 'Commission Income CB', 'Commission Income CB SIP',
                         'Commission Income JKO',
                         'Commission Income Local', 'Custom_Tax_CB', 'HL/OK價差', 'Local ASF', 'Local SIP',
                         'Others CB', 'Others Local', 'Rebate CB', 'Rebate JKO', 'Rebate Local',
                         'SFR CB', 'SFR Local', 'SIP Margin CB', 'Seller Coin Cashback Voucher CB',
                         'Seller Coin Cashback Voucher Local',
                         'Service fee revenue CB', 'Service fee revenue JKO', 'Service fee revenue Local', 'Shopee Coin CB',
                         'Shopee Coin JKO',
                         'Shopee Coin Local', 'Transaction_Fee CB', 'Transaction_Fee JKO', 'Transaction_Fee Local',
                         'Voucher CB',
                         'Voucher JKO', 'Voucher Local', '收款實現運費 Non HL/OK', '逆物流 CB', '逆物流 Local',
                         '逆物流運費 JKO', 'JKO Others'
                         , '711/FM價差']

        result = df.query("tag.isin(['RECON']) and mark.isin(@recon_details)").pivot_table(
            values='amt',
            index=['mark'],
            columns='period',
            aggfunc=np.sum
        ).fillna(0).applymap(lambda x: format(int(x), ',')).reset_index()
        # .reset_index()#.fillna(0).applymap(lambda x: format(int(x), ','))
        return result

