import pandas as pd
import shap

import numpy as np
import utils


def create_base(model, df, inference_name, model_id, target, model_type):
    target_df = df.pop(target)
    target_df = pd.DataFrame(target_df)

    importance = create_importance(df, model)
    feature_di, feature_cnt, feature_labels = create_range_count(df)
    feature_rc = merge_data(feature_di, feature_cnt, feature_labels, importance)
    target_di, target_cnt, target_label = create_range_count(target_df, model_type)
    target_rc = merge_data(target_di, target_cnt, target_label, None)

    base_data = {"target": target_rc, "features": feature_rc}
    result, message = utils.save_data('trainingdata_base_metric', f"{inference_name}_{model_id}", base_data)
    if result is False:
        return result, message

    return result, message


def create_importance(df, model):
    length = df.columns.__len__()
    df_length = len(df)
    df_columns = df.columns
    slice_value = 100 if df_length > 100 else df_length
    explainer = shap.explainers.Permutation(model.predict, df)

    # 중요도 체크를 위한 샘플링
    df_sample = df.sample(n=slice_value, random_state=1004)

    shap_values = explainer(df_sample)

    value_list = [0] * length
    cnt = 0

    for v in shap_values.values:
        cnt += 1
        for i in range(len(v)):
            value_list[i] += abs(v[i])

    for i in range(length):
        value_list[i] = round(value_list[i] / cnt, 3)

    j = []
    MIN = min(value_list)
    MAX = max(value_list)

    for i in range(length):
        temp = (value_list[i] - MIN) / (MAX - MIN)
        j.append(temp)

    imp = np.round(j, 3)

    importance = {}
    for n in range(length):
        importance[df_columns[n]] = imp[n]

    return importance


def create_range_count(df, model_type=False):
    di = {}

    labels, _, min_data, max_data, step = utils.get_feature_info(df)

    for i in range(len(labels)):
        temp = []
        for j in range(10):
            temp.append(min_data[i] + float(step[i] * j))
        temp.append(float(max_data[i]))
        di[labels[i]] = temp

    cnt = get_count(di, df, "training")

    return di, cnt, labels


def get_count(range_data, data_list, data_type):
    temp = {}
    if data_type == "training":
        for i in range_data:
            cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for value in data_list[i]:
                for n in range(0, 10):
                    if n == 9:
                        if range_data[i][n] <= value <= range_data[i][n + 1]:
                            cnt[n] += 1
                    else:
                        if range_data[i][n] <= value < range_data[i][n + 1]:
                            cnt[n] += 1
                            break
            temp[i] = cnt
    elif data_type == 'inference':
        for i in range_data:
            cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for value in data_list[i]:
                if type(value) != float and type(value) != int:
                    cnt[12] += 1
                else:
                    if value < range_data[i]['range'][0]:
                        cnt[0] += 1
                    elif range_data[i]['range'][10] < value:
                        cnt[11] += 1
                    elif range_data[i]['range'][9] <= value <= range_data[i]['range'][10]:
                        cnt[10] += 1
                    else:
                        for n in range(0, 10):
                            if range_data[i]['range'][n] <= value < range_data[i]['range'][n + 1]:
                                cnt[n + 1] += 1
                                break

            cnt = {str(i): cnt[i] for i in range(len(cnt))}
            temp[i] = cnt

    return temp


def merge_data(di, cnt, labels, imp):
    rc = {}
    if di is None:
        for i in range(len(labels)):
            rc[labels[i]] = cnt[labels[i]]
        return rc
    else:
        percent, _ = utils.get_percent(cnt)
        if imp is None:
            for label in labels:
                rc[label] = {
                    "range": di[label],
                    "count": cnt[label],
                    "percent": percent[label],
                    "importance": 1
                }
        else:
            for label in labels:
                rc[label] = {
                    "range": di[label],
                    "count": cnt[label],
                    "percent": percent[label],
                    "importance": imp[label]
                }

        return rc
