from utils import LoadData, msearch_data, get_percent, search_aggregations, human_format

import time
import numpy as np
from collections import defaultdict
from dateutil import parser
import math


def load_training(inference_name):
    info_data = LoadData("trainingdata_feature_info", inference_name)
    _, info = info_data.load_data(True)
    # 먼저 테스트 데이터셋의 분포도 불러옴
    base_data = LoadData('trainingdata_base_metric', inference_name)
    _, base = base_data.load_data(True)
    target_base = base['target']
    feature_base = base['features']
    return target_base, feature_base, info


def get_feature_drift_dict(inference_name, model_id, start_time, end_time, drift_threshold, importance_threshold):
    label_dict, total_dict, imp, cnt = feature_detail(f"{inference_name}_{model_id}", start_time, end_time)
    if label_dict is False:
        return False, total_dict, cnt
    psi_score_dict = feature_drift(label_dict, total_dict)
    drift_threshold = drift_threshold
    importance_threshold = importance_threshold
    drift_dict = tran_drift_dict(psi_score_dict, imp, drift_threshold, importance_threshold)
    return True, drift_dict, cnt


def tran_drift_dict(psi_score_dict, importance, drift_threshold, importance_threshold):
    index = 0
    drift_score = {}
    feature_importance = {}
    labels = {}

    for label in psi_score_dict:
        drift_score[index] = psi_score_dict[label]
        feature_importance[index] = importance[label]
        labels[index] = label
        index += 1

    drift_dict = {"drift score": drift_score, "feature importance": feature_importance, "Label": labels,
                  "threshold": {"drift": drift_threshold, "importance": importance_threshold}}

    return drift_dict


def feature_drift(label_dict, total_dict):
    psi_score_dict = {}
    for label in label_dict.values():
        train_percent = total_dict[label]['Training']['Percentage']
        scoring_percent = total_dict[label]['Scoring']['Percentage']
        train_percent = list(train_percent.values())
        scoring_percent = list(scoring_percent.values())

        psi_value = 0
        for i in range(len(train_percent)):
            # percent 값이 0일 경우 보정수치인 0.0001 (0.01%) 를 넣어준다
            scoring_value = scoring_percent[i]
            if scoring_value == 0:
                scoring_value = 0.0001

            train_value = train_percent[i]
            if train_value == 0:
                train_value = 0.0001

            psi_value += (scoring_value - train_value) * math.log(scoring_value / train_value)
        psi_score_dict[label] = psi_value

    return psi_score_dict


def feature_detail(inference_name, start_time, end_time):
    try:
        target_base, feature_base, info = load_training(inference_name)
    except Exception:
        return False, f"NotFoundError : Traindata or Feature Info Not Found. InferenceName: {inference_name}", 0, 0

    base_target_percent, base_target_range, base_target_imp = filter_value(target_base)
    base_feature_percent, base_feature_range, base_feature_imp = filter_value(feature_base)
    top_imp_label = top_imp(base_feature_imp)

    # 엘라스틱에서 시간범위 내의 피쳐의 카운터를 불러옴
    # feature_index = tuple(info['feature index'].values())
    feature_index = tuple(top_imp_label)
    target_index = tuple(info['target'].keys())

    result, feature_value = msearch_data(feature_index, f"data_drift_{inference_name}", start_time, end_time)
    if result is False:
        return False, feature_value, 0, 0
    feature_value = sort_value(feature_value)

    result, target_value = msearch_data(target_index, f"data_drift_{inference_name}", start_time, end_time)
    if result is False:
        return False, target_value, 0, 0
    target_value = sort_value(target_value)

    # 피쳐카운터로 분포도 계산함
    feature_percent, feature_cnt = get_percent(feature_value)
    target_percent, _ = get_percent(target_value)

    # 타겟과 피쳐데이터 결합
    base_percent = merge_target_feature(base_target_percent, base_feature_percent)
    base_range = merge_target_feature(base_target_range, base_feature_range)
    base_imp = merge_target_feature(base_target_imp, base_feature_imp)
    scoring_percent = merge_target_feature(target_percent, feature_percent)

    # 라벨
    Label = make_label(target_index, feature_index)
    train_dict = merge_training_scoring('Training', base_range, base_percent)
    score_dict = merge_training_scoring('Scoring', base_range, scoring_percent)

    # 피쳐 디테일 완성
    total_dict = mg(train_dict, score_dict)
    label_dict = {i: Label[i] for i in range(len(Label))}

    return label_dict, total_dict, base_imp, feature_cnt


def get_feature_detail_dict(inference_name, model_id, start_time, end_time):
    label_dict, total_dict, imp, cnt = feature_detail(f"{inference_name}_{model_id}", start_time, end_time)
    if label_dict is False:
        return False, total_dict, 0
    feature_detail_dict = {"Label": label_dict, "features": total_dict}
    return True, feature_detail_dict, cnt


def sort_value(value):
    value_list = {}
    for v in value:
        temp = list(range(13))
        for i in value[v]:
            temp[int(i)] = value[v][i]['value']
        value_list[v] = temp

    return value_list


def filter_value(value):
    percent = {}
    range_dic = {}
    imp = {}
    for i in value:
        # percent[i] = make_percent(value[i]['percent'])
        percent[i] = value[i]['percent']
        range_dic[i] = value[i]['range']
        imp[i] = value[i]['importance']

    return percent, range_dic, imp


def top_imp(imp):
    sort_imp = sorted(imp.items(), key=lambda item: item[1], reverse=True)
    sort_imp = sort_imp[:24]
    label = []
    for i in range(len(sort_imp)):
        label.append(sort_imp[i][0])

    return label


def make_label(target, feature):
    return target + feature


def merge_training_scoring(key, range_dic, percent):
    merge_dict = {}
    labels = {}
    percentage = {}

    for label in range_dic:
        n = 2
        checker = False
        breaker = False
        base_dict = defaultdict(dict)
        while not checker:
            breaker = False
            tem = np.round(range_dic[label], n)
            for i in range(len(tem) - 1):
                if tem[i] == tem[i + 1]:
                    n += 1
                    breaker = True
                    break
                if breaker is False:
                    range_dic[label] = tem
                    checker = True
            time.sleep(0.001)
        label_index = 0
        percent_index = 0
        labels[0] = f"<{human_format(range_dic[label][0])}"
        for j in range_dic[label]:
            labels[label_index + 1] = human_format(j)
            label_index += 1
        labels[11] = f">{human_format(range_dic[label][10])}"
        labels[12] = "missing"
        if key == "Training":
            percentage[0] = 0.0
            for v in percent[label]:
                percentage[percent_index + 1] = v
                percent_index += 1
            percentage[11] = 0.0
            percentage[12] = 0.0
        else:
            for v in percent[label]:
                percentage[percent_index] = v
                percent_index += 1

        base_dict[key]['Percentage'] = percentage.copy()
        base_dict[key]["Label"] = labels.copy()
        merge_dict.setdefault(label, {})
        merge_dict[label].update(base_dict)
    return merge_dict


def merge_target_feature(target, feature):
    merge_dict = dict(target, **feature)
    return merge_dict


def mg(dic1, dic2):
    for i in dic1:
        dic1[i]['Scoring'] = dic2[i]['Scoring']
    return dic1


def get_percent_over_time(inference_name, model_history_id, start_time, end_time):
    start_time, end_time, interval, offset = get_interval(start_time, end_time)

    # history 불러와서 시간 비교 해야함.
    model_history = LoadData(f"data_drift_model_history_{inference_name}", model_history_id)
    result, history = model_history.load_data(True)
    if result is True:
        created = history['created']
        deleted = history['deleted']

        start_parse = parser.parse(start_time)
        end_parse = parser.parse(end_time)
        created_parse = parser.parse(created)

        if end_parse < created_parse:
            return False, f"No data between {start_time} - {end_time}"
        else:
            if start_parse < created_parse:
                start_time = created
            if deleted != "None":
                deleted_parse = parser.parse(deleted)
                if end_parse > deleted_parse:
                    end_time = deleted

    query = {
        "range": {
            "timestamp": {
                "gte": f"{start_time}",
                "lt": f"{end_time}"
            }
        }
    }
    aggs = {
        "pred_percentiles": {
            "percentiles": {
                "field": "predictions",
                "percents": [10, 90]
            }
        },
        "pred_stats": {
            "stats": {
                "field": "predictions"
            }
        }
    }
    date_aggs = {
        "timeline": {
            "date_histogram": {
                "field": "timestamp",
                "fixed_interval": interval,
                "extended_bounds": {
                    "min": start_time,
                    "max": end_time
                },
                "offset": offset
            },
            "aggs": aggs
        }
    }

    result, data = search_aggregations(f"inference_org_{inference_name}", query, date_aggs)
    if result is False:
        return result, data

    value = []
    doc_count = 0
    date_list = []
    for v in data['timeline']['buckets']:
        date_list.append(v['key_as_string'])
        if v['doc_count'] != 0:
            doc_count = v['doc_count']
        value.append({
            "count": v['pred_stats']['count'] if v['pred_stats']['count'] != 0 else np.nan,
            "avg": v['pred_stats']['avg'] if v['pred_stats']['avg'] is not None else np.nan,
            "10th": v['pred_percentiles']['values']['10.0'] if v['pred_percentiles']['values'][
                                                                   '10.0'] is not None else np.nan,
            "90th": v['pred_percentiles']['values']['90.0'] if v['pred_percentiles']['values'][
                                                                   '90.0'] is not None else np.nan
        })

    if doc_count == 0:
        return False, f"No data between {start_time} - {end_time}"

    data = {
        "data": value,
        "date": date_list
    }

    return True, data


def get_interval(start_time, end_time):
    start_parse = parser.parse(start_time)
    start_time = start_parse.strftime('%Y-%m-%dT%H:00:00.000Z')
    start_hour = start_parse.hour
    end_parse = parser.parse(end_time)
    end_time = end_parse.strftime('%Y-%m-%dT%H:00:00.000Z')

    term = end_parse - start_parse
    term_day = term.days
    if term_day < 7:
        return start_time, end_time, '1h', "+0h"
    elif 7 <= term_day < 60:
        return start_time, end_time, '1d', f"+{start_hour}h"
    elif 60 <= term_day < 730:
        return start_time, end_time, '7d', f"+{start_hour}h"
    else:
        return start_time, end_time, '30d', f"+{start_hour}h"
