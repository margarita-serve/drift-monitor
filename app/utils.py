import logging
import os
import base64
import time
from minio import Minio
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, exceptions
from functools import lru_cache
import shutil
from secrets import token_hex
from datetime import datetime

from tensorflow import keras
import torch
import pickle
import joblib
import lightgbm
import xgboost

KSERVE_API_DEFAULT_STORAGE_ENDPOINT = os.environ.get('KSERVE_API_DEFAULT_STORAGE_ENDPOINT')
KSERVE_API_DEFAULT_DATABASE_ENDPOINT = os.environ.get('KSERVE_API_DEFAULT_DATABASE_ENDPOINT')
KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID = os.environ.get('KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID')
KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY = os.environ.get('KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY')

ES = Elasticsearch(KSERVE_API_DEFAULT_DATABASE_ENDPOINT)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()


# 경로를 잘라줌
def split_path(data):
    bucket_name = data[:data.find('/')]
    path = data[data.find('/') + 1:]

    return [bucket_name, path]


def decode_key(value):
    b_bytes = value.encode('ascii')
    m_bytes = base64.b64decode(b_bytes)
    decode_value = m_bytes.decode('ascii')
    return decode_value


def minio_client(path):
    sp = split_path(path)
    access_key = decode_key(KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID)
    secret_key = decode_key(KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY)
    client = Minio(KSERVE_API_DEFAULT_STORAGE_ENDPOINT, access_key, secret_key, secure=False)

    path_token = token_hex(8)
    for item in client.list_objects(sp[0], recursive=True):
        if sp[1] in item.object_name:
            client.fget_object(sp[0], item.object_name, f"temp/{path_token}/{item.object_name}")
    path = f"temp/{path_token}/{sp[1]}"
    return path


def make_index(index_name):
    if not ES.indices.exists(index=index_name):
        mapping = {"settings": {'mapping': {'ignore_malformed': True}}}
        ES.indices.create(index=index_name, body=mapping)
        logger.info(f"Success create Index : {index_name}")


def get_feature_info(df):
    labels = df.columns.tolist()
    types = df.dtypes.tolist()
    types = list(map(str, types))
    min_data = df.min()
    max_data = df.max()
    step = (max_data - min_data) / 10

    return labels, types, min_data, max_data, step


def get_round(value, point):
    if type(point) == list:
        for i in range(len(point)):
            value[i] = round(value[i], point[i])
    else:
        value = list(np.round(value, point))
    return value


def save_data(index, name, document):
    make_index(index)
    try:
        ES.index(index=index, id=name, document=document)
        logger.info(f"Success input data in Index: {index} ID: {name}")
        return True, "Success input data in Index: {index} ID: {name}"

    except Exception as err:
        logger.exception(f"Error: {err}\n")
        return False, err


def upsert_data(index, name, document, query):
    try:
        ES.update(index=index, id=name, body={"script": {
            "source": query,
            "lang": "painless"
        }, 'upsert': document})
        logger.info(f"index: {index}, id: {name} upsert")
        return True, f"index: {index}, id: {name} upsert"
    except Exception as err:
        logger.exception(f'Error : {err}\n')
        return False, err


def update_data(index, name, document):
    try:
        ES.update(index=index, id=name, doc=document)
        return True, f"index: {index}, id: {name} update"
    except Exception as err:
        return False, err


def delete_id_data(index, name):
    try:
        ES.delete(index=index, id=name)
    except exceptions.ConnectionError as err:
        logger.exception(f"Connection Error : {err.message}")
    except exceptions.NotFoundError as err:
        logger.exception(f"NotFound Error : {err.message}")
    except Exception:
        logger.exception(f"Failed to delete id: {name} for index: {index}")


def delete_index_data(index):
    try:
        ES.indices.delete(index=index)
    except exceptions.ConnectionError as err:
        logger.exception(f"Connection Error : {err.message}")
    except exceptions.NotFoundError as err:
        logger.exception(f"NotFound Error : {err.message}")
    except Exception:
        logger.exception(f"Failed to delete index: {index}")


class LoadData:
    def __init__(self, index, inference_id):
        self.index = index
        self.inference_id = inference_id

    @lru_cache(maxsize=32)
    def load_data(self, log_print):

        try:
            value = ES.get(index=self.index, id=self.inference_id)['_source']
            return True, value

        except exceptions.ConnectionTimeout as err:
            if log_print is True:
                logger.exception(err)
            return False, err

        except Exception as err:
            if log_print is True:
                logger.exception(f'Error : {err}\n')
            return False, err


def search_scroll_index(index, query=False):
    try:
        if query is False:
            items = ES.search(index=index, scroll='30s', size=100)
        else:
            items = ES.search(index=index, query=query, scroll='30s', size=100)

        sid = items['_scroll_id']
        fetched = items['hits']['hits']
        total = []

        for i in fetched:
            total.append(i)
        while len(fetched) > 0:
            items = ES.scroll(scroll_id=sid, scroll='30s')
            fetched = items['hits']['hits']
            for i in fetched:
                total.append(i)
            time.sleep(0.001)

        return True, total
    except exceptions.BadRequestError as err:
        return False, err
    except Exception as err:
        return False, err


def search_aggregations(index, query, aggs):
    try:
        items = ES.search(index=index, query=query, aggs=aggs, request_timeout=300, size=0)
        return True, items['aggregations']
    except exceptions.BadRequestError as err:
        return False, err
    except Exception as err:
        return False, err


# def search_data(labels, index, start_time, end_time):
#     value = {}
#     query = get_query(start_time, end_time)
#     try:
#         for i in labels:
#             aggs = get_aggs(i)
#             items = ES.search(index=index, query=query, aggs=aggs, size=0)
#             if items['hits']['total']['value'] == 0:
#                 return False, f'ID: {index} No data between {start_time} ~ {end_time} '
#             else:
#                 value[i] = items['aggregations']
#         return True, value
#     except exceptions.NotFoundError as err:
#         return False, err
#     except Exception as err:
#         logger.exception(f'search data func Error : {err}\n')
#         return False, err


def msearch_data(labels, index, start_time, end_time):
    value = {}
    query = get_query(start_time, end_time)
    mquery = []
    try:
        for i in labels:
            aggs = get_aggs(i)
            mquery.append({'index': index})
            mquery.append({"query": query, "aggs": aggs})
        result, values = msearch_query(mquery)
        if result is False:
            return False, values
        responses = values['responses']
        for i in range(len(responses)):
            if responses[i]['status'] == 404:
                return False, f"no such index {index}"
            elif responses[i]['hits']['total']['value'] == 0:
                return False, f'No data between {start_time} ~ {end_time} '
            value[labels[i]] = responses[i]['aggregations']
        return True, value
    except Exception as err:
        logger.exception(f"search data func Error : {err}\n")
        return False, err


def get_query(start_time, end_time):
    q = {
        "range": {
            "date": {
                "gte": f"{start_time}",
                "lt": f"{end_time}"
            }
        }
    }
    return q


def get_aggs(feature_name):
    aggs = {
        "0": {
            "sum": {
                "field": f"{feature_name}.0"
            }
        },
        "1": {
            "sum": {
                "field": f"{feature_name}.1"
            }
        },
        "2": {
            "sum": {
                "field": f"{feature_name}.2"
            }
        },
        "3": {
            "sum": {
                "field": f"{feature_name}.3"
            }
        },
        "4": {
            "sum": {
                "field": f"{feature_name}.4"
            }
        },
        "5": {
            "sum": {
                "field": f"{feature_name}.5"
            }
        },
        "6": {
            "sum": {
                "field": f"{feature_name}.6"
            }
        },
        "7": {
            "sum": {
                "field": f"{feature_name}.7"
            }
        },
        "8": {
            "sum": {
                "field": f"{feature_name}.8"
            }
        },
        "9": {
            "sum": {
                "field": f"{feature_name}.9"
            }
        },
        "10": {
            "sum": {
                "field": f"{feature_name}.10"
            }
        },
        "11": {
            "sum": {
                "field": f"{feature_name}.11"
            }
        },
        "12": {
            "sum": {
                "field": f"{feature_name}.12"
            }
        }
    }
    return aggs


def get_percent(cnt):
    percent = {}
    total = 0
    for i in cnt:
        total = sum(cnt[i])
        temp = []
        for j in cnt[i]:
            if total == 0:
                temp.append(0)
            else:
                temp.append(round((j / total), 3))
        percent[i] = temp
    return percent, total


def delete_file(path):
    paths = path.split('/')
    try:
        shutil.rmtree(f"{paths[0]}/{paths[1]}")
    except Exception as err:
        logger.exception(err)


def create_indices(index):
    try:
        ES.indices.create(index=index)
        return f'Create index: {index}'
    except exceptions.BadRequestError as err:
        return err
    except Exception as err:
        return err


def check_index(inference_name, current_model):
    try:
        status = ES.indices.exists(index=f"{inference_name}_{current_model}")
        return True, status
    except Exception as err:
        return False, err


def read_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    if "Unnamed: 0" in df.columns:
        df.pop("Unnamed: 0")

    delete_file(dataset_path)
    logger.info(f"delete {dataset_path}")
    return df


def load_model(framework, path):
    try:
        if framework == 'TensorFlow':
            model = keras.models.load_model(path)
        elif framework == 'PyTorch':
            model = torch.load(f=path)
        elif framework == 'SkLearn':
            model = joblib.load(path)
        elif framework == 'XGBoost':
            model = xgboost.Booster(model_file=path)
        else:
            model = lightgbm.Booster(model_file=path)
    except Exception:
        try:
            model = pickle.load(open(path, 'rb'))
        except Exception:
            delete_file(path)
            logger.info(f"delete {path}")
            return False

    delete_file(path)
    logger.info(f"delete {path}")
    return model


def msearch_query(query):
    try:
        res = ES.msearch(
            body=query
        )
        return True, res
    except Exception as err:
        return False, err


def convertTimestamp(timeString):
    times = datetime.strptime(timeString, "%Y-%m-%d:%H")
    convTime = times.strftime("%Y-%m-%dT%H:%M:%SZ")

    return convTime


def validate(model, df, target, framwork):
    # 1. 데이터 프레임 타입확인
    for i in df.dtypes:
        if i == float or i == int:
            pass
        else:
            # todo 추후 string 등 추가 예정 현재는 int 와 float만 받음
            return False, "DataSet Type Error : Column data type must be int or float."
    # 2. None value check
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        return False, "DataSet Value Error : The value of the dataset has null."
    # 3. target column check
    if target in df.columns:
        df = df.drop(columns=target)
    else:
        return False, f"DataSet Column Error : Target label :{target} does not exist in dataset column."
    # 4. sample predict
    sample = df.sample(10)
    try:
        if framwork == "PyTorch":
            df_tensor = torch.from_numpy(sample.values)
            df_tensor = torch.tensor(df_tensor, dtype=torch.float32)
            model(df_tensor)
        else:
            model.predict(sample)
    except Exception as err:
        return False, f"Model Predict Error : {err}"

    return True, "Validate Check Pass"


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def make_percent(value):
    for i in range(len(value)):
        value[i] = round(value[i] / 100, 3)
    return value
