import logging
import os
import time
from kafka import KafkaConsumer, KafkaProducer
from json import loads, dumps
import utils
from utils import LoadData
from create_feature_base import get_count, merge_data
import pandas as pd
import numpy as np
from dateutil import parser

KSERVE_API_DEFAULT_KAFKA_ENDPOINT = os.environ.get('KSERVE_API_DEFAULT_KAFKA_ENDPOINT')

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def makeproducer():
    try:
        producer = KafkaProducer(
            acks=1,
            compression_type='gzip',
            bootstrap_servers=[KSERVE_API_DEFAULT_KAFKA_ENDPOINT],
            value_serializer=lambda x: dumps(x).encode('utf-8')
        )
        return producer
    except Exception as err:
        logger.exception(err)
        time.sleep(60)
        makeproducer()


def produceKafka(producer, message):
    producer.send('datadrift-monitoring-data', value=message)
    producer.flush()


def consumeKafka():
    # topic, broker list
    try:
        consumer = KafkaConsumer(
            'logstash-inference-data',
            bootstrap_servers=[KSERVE_API_DEFAULT_KAFKA_ENDPOINT],
            group_id='datadrift-monitor',
            auto_offset_reset='latest',
            enable_auto_commit=True,
            # consumer_timeout_ms=1000,
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
    except Exception as err:
        logger.exception(err)
        time.sleep(60)
        consumeKafka()

    # 접속확인
    if consumer.bootstrap_connected() is True:
        logger.info('Kafka consumer is running!')

    for message in consumer:
        if "tags" in message.value:
            logger.warning(message.value['tags'])
        else:
            try:
                result, message = trans_data(message.value)
                if result is False:
                    raise Exception(message)
            except Exception as err:
                logger.exception(f'Kafka consumer error: {err}')


def trans_data(value):
    timestamp = value['timestamp']
    parser_date = parser.parse(timestamp)
    hour_date = parser_date.strftime('%Y-%m-%dT%H:00:00')
    inference_id = value['inference_servicename']
    event_type = value['type']
    monitor_data = LoadData('data_drift_monitor_setting', inference_id)
    result, monitor = monitor_data.load_data(True)
    if result is False:
        return False, monitor
    if monitor['status'] == 'disable':
        return True, 'disable'
    model_id = monitor['current_model']
    load_data = LoadData("trainingdata_feature_info", f"{inference_id}_{model_id}")
    result, info = load_data.load_data(True)
    if result is False:
        return False, info
    load_base = LoadData("trainingdata_base_metric", f"{inference_id}_{model_id}")
    result, base_data = load_base.load_data(True)
    if result is False:
        return False, base_data
    index = info['feature_index']

    if event_type == 'request':
        instances = value['instances']
        temp = {}
        target = list(info['target'].keys())
        range_data = base_data['features']
        if len(np.shape(instances)) == 1:
            for i in index:
                temp_list = []
                for v in instances:
                    try:
                        temp_list.append(v)
                    except Exception as err:
                        return False, err
                temp[index[i]] = temp_list
            df = pd.DataFrame(temp)
        else:
            df = pd.DataFrame(instances, columns=index.values())

        index = list(index.values())
        cnt = get_count(range_data, df, 'inference')
        doc = merge_data(None, cnt, index, None)
        upsert_query = create_upsert_data(doc)
        doc[target[0]] = {str(i): 0 for i in range(13)}
        doc["date"] = hour_date
        result, message = utils.upsert_data(f"data_drift_{inference_id}_{model_id}", hour_date, doc, upsert_query)
        if result is False:
            return False, message

    elif event_type == 'response':
        predictions = value['predictions']
        label = list(info['target'].keys())
        range_data = base_data['target']
        temp = {}
        temp_list = []
        if len(np.shape(predictions)) == 1:
            for v in predictions:
                try:
                    if len(np.shape(predictions)) == 1:
                        temp_list.append(v)
                    else:
                        temp_list.append(v[0])
                except Exception as err:
                    return False, err
            temp[label[0]] = temp_list
            df = pd.DataFrame(temp)
        else:
            df = pd.DataFrame(predictions, label)
        cnt = get_count(range_data, df, 'inference')
        doc = merge_data(None, cnt, label, None)
        upsert_query = create_upsert_data(doc)
        index = list(index.values())
        for i in index:
            doc[i] = {str(i): 0 for i in range(13)}
        doc["date"] = hour_date
        result, message = utils.upsert_data(f"data_drift_{inference_id}_{model_id}", hour_date, doc, upsert_query)
        if result is False:
            return False, message
    return True, 'Success'


def create_upsert_data(doc):
    filter_dict = {}
    query = ''
    for label in doc:
        target = doc[label]
        filter_value = dict(filter(lambda elem: elem[1] != 0, target.items()))
        filter_dict[label] = filter_value.copy()
        for key in filter_dict[label]:
            query += f"ctx._source['{label}']['{key}']+={filter_dict[label][key]};"
    return query
