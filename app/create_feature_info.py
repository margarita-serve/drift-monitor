import datetime

import utils


def create_info(df, inference_name, model_id, target, model_type, multiclass_target_classes=False):
    labels, types, min_data, max_data, step = utils.get_feature_info(df)

    info = {"feature_index": {}, "target": {}, "features": {}, "model_type": model_type}

    j = 0
    for i in range(len(labels)):
        if labels[i] != target:
            info['feature_index'][i - j] = labels[i]
            info['features'][labels[i]] = {"type": types[i], "min": min_data[i], "max": max_data[i],
                                           "step": step[i]}
        else:
            j = 1
            if model_type == 'Multiclass':
                info['target'] = {
                    target: {"type": types[i], "classes": {v: k for v, k in enumerate(multiclass_target_classes)}}}
            else:
                info['target'] = {target: {"type": types[i], "min": min_data[i], "max": max_data[i], "step": step[i]}}

    result, message = utils.save_data('trainingdata_feature_info', f"{inference_name}_{model_id}", info)
    if result is False:
        return False, message
    else:
        return True, message


def create_default_monitor_setting(inference_name, model_id, drift_threshold, importance_threshold, monitor_range,
                                   low_imp_atrisk_count, low_imp_failing_count, high_imp_atrisk_count,
                                   high_imp_failing_count):
    setting = {
        "current_model": model_id,
        "drift_threshold": drift_threshold,
        "importance_threshold": importance_threshold,
        "monitor_range": monitor_range,
        "low_imp_atrisk_count": low_imp_atrisk_count,
        "low_imp_failing_count": low_imp_failing_count,
        "high_imp_atrisk_count": high_imp_atrisk_count,
        "high_imp_failing_count": high_imp_failing_count,
        "status": "enable"
    }
    result, message = utils.save_data('data_drift_monitor_setting', inference_name, setting)
    if result is False:
        return False, message
    else:
        return True, message


def add_history(inference_name, model_id):
    monitor_setting_data = utils.LoadData("data_drift_monitor_setting", inference_name)
    result, monitor_setting = monitor_setting_data.load_data(False)
    now_date = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    if result is True:
        # 셋팅이 있다 = 모델 replace 이다 그러면 history에서 전 모델의 종료시간을 찍고 history 생성
        pre_model_id = monitor_setting['current_model']
        doc = {
            "deleted": now_date
        }
        result, message = utils.update_data(f'data_drift_model_history_{inference_name}', pre_model_id, doc)
        if result is False:
            return False, message

    doc = {
        "created": now_date,
        "deleted": "None"
    }
    result, message = utils.save_data(f'data_drift_model_history_{inference_name}', model_id, doc)
    if result is False:
        return False, message
    else:
        return True, message
