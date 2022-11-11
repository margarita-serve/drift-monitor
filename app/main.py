import time
from multiprocessing import Process
import logging
import os

from flask import Flask
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse

import utils
from kafka_func import consumeKafka, produceKafka, makeproducer
from create_feature_info import create_info, create_default_monitor_setting, add_history
from create_feature_base import create_base
from cal_drift import get_feature_drift_dict, get_feature_detail_dict, get_percent_over_time

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='KoreServe Drift Monitor Service API')

ns = api.namespace('', description='REST-API operations')

#######################################################################
# restX Input Model
#######################################################################

base_monitor_model = api.model('BaseMonitor', {
    "train_dataset_path": fields.String(example='dataset/train_data.csv', required=True),
    "model_path": fields.String(example='testmodel/mpg2/1', required=True),
    "inference_name": fields.String(example="mpg-sample", required=True),
    "model_id": fields.String(example="000001", required=True),
    "target_label": fields.String(example="MPG", required=True),
    "model_type": fields.String(enum=['Regression', 'Binary'], example="Regression", required=True),
    "framework": fields.String(enum=['TensorFlow', 'SkLearn', 'XGBoost', 'LightGBM', 'PyTorch'],
                               example="TensorFlow", required=True),
    "drift_threshold": fields.Float(example=0.15, required=True),
    "importance_threshold": fields.Float(example=0.5, required=True),
    "monitor_range": fields.String(example='7d', required=True),
    "low_imp_atrisk_count": fields.Integer(example=1, required=True),
    "low_imp_failing_count": fields.Integer(example=0, required=True),
    "high_imp_atrisk_count": fields.Integer(example=0, required=True),
    "high_imp_failing_count": fields.Integer(example=1, required=True),
})

monitor_setting_model = api.model('MonitorSetting', {
    "current_model": fields.String(example='000001'),
    "drift_threshold": fields.Float(example=0.15),
    "importance_threshold": fields.Float(example=0.5),
    "monitor_range": fields.String(example='7d'),
    "low_imp_atrisk_count": fields.Integer(example=1),
    "low_imp_failing_count": fields.Integer(example=0),
    "high_imp_atrisk_count": fields.Integer(example=0),
    "high_imp_failing_count": fields.Integer(example=1)
})

#######################################################################
# restX Output Model
#######################################################################

base_output_model = api.model("BaseOutputModel", {
    "message": fields.String,
    "inference_name": fields.String
})

feature_output_model = api.model("FeatureDetailOutputModel", {
    "message": fields.String,
    "data": fields.String,
    "start_time": fields.String,
    "end_time": fields.String,
    "prediction_count": fields.Integer
})

drift_metrics_output_model = api.model("DriftMetricsOutputModel", {
    "message": fields.String,
    "data": fields.String,
    "start_time": fields.String,
    "end_time": fields.String
})


############################################################
# HTTP Routing
############################################################

@ns.route("/drift-monitor/feature-detail/<string:inferencename>")
@ns.param('end_time', 'example=2022-05-13:05', required=True)
@ns.param('start_time', 'example=2022-05-04:05', required=True)
@ns.param('model_history_id', 'Model History ID', required=True)
@ns.param('inferencename', 'Kserve Inferencename')
class FeatureDetailAPI(Resource):
    @ns.marshal_with(feature_output_model, code=200, skip_none=True)
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('start_time', required=True, type=str,
                            location='args', help='2022-05-04:05')
        parser.add_argument('end_time', required=True, type=str,
                            location='args', help='2022-05-13:05')
        parser.add_argument('model_history_id', required=True, type=str,
                            location='args', help='000001')

        args = parser.parse_args()

        start_time, end_time, model_history_id = parsing_get_data(args)
        try:
            start_time = utils.convertTimestamp(start_time)
            end_time = utils.convertTimestamp(end_time)
        except:
            return {"message": "time parser error, time format must be yyyy-mm-dd:hh"}, 400
        # monitor_setting = get_monitor_settings(inference_name)
        model_id = model_history_id
        result, feature_detail_dict, predict_count = get_feature_detail_dict(inferencename, model_id, start_time,
                                                                             end_time)
        if result is False:
            if str(feature_detail_dict).find("NotFoundError") >= 0:
                return {"message": feature_detail_dict, "inference_name": inferencename}, 404
            elif str(feature_detail_dict).find("No data") >= 0:
                return {"message": feature_detail_dict, "inference_name": inferencename,
                        "start_time": start_time, "end_time": end_time, "data": [],
                        "prediction_count": 0}, 200
            return {"message": feature_detail_dict, "inference_name": inferencename}, 400

        return {"message": "Success get feature detail data", "inference_name": inferencename,
                "start_time": start_time, "end_time": end_time, "data": feature_detail_dict,
                "prediction_count": predict_count}, 200


@ns.route("/drift-monitor/feature-drift/<string:inferencename>")
@ns.param('end_time', 'example=2022-05-13:05', required=True)
@ns.param('start_time', 'example=2022-05-04:05', required=True)
@ns.param('model_history_id', 'Model History ID', required=True)
@ns.param('inferencename', 'Kserve Inferencename')
class FeatureDriftAPI(Resource):
    @ns.marshal_with(feature_output_model, code=200, skip_none=True)
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('start_time', required=True, type=str,
                            location='args', help='2022-05-04:01')
        parser.add_argument('end_time', required=True, type=str,
                            location='args', help='2022-05-13:01')
        parser.add_argument('model_history_id', required=True, type=str,
                            location='args', help='000001')

        args = parser.parse_args()

        start_time, end_time, model_history_id = parsing_get_data(args)
        try:
            start_time = utils.convertTimestamp(start_time)
            end_time = utils.convertTimestamp(end_time)
        except:
            return {"message": "time parser error, time format must be yyyy-mm-dd:hh"}, 400
        result, monitor_setting = get_monitor_settings(inferencename)
        if result is False:
            return {"message": monitor_setting, "inference_name": inferencename}, 400
        model_id = model_history_id
        drift_threshold = monitor_setting['drift_threshold']
        importance_threshold = monitor_setting['importance_threshold']

        result, feature_drift_dict, predict_count = get_feature_drift_dict(inferencename, model_id, start_time,
                                                                           end_time, drift_threshold,
                                                                           importance_threshold)
        if result is False:
            if str(feature_drift_dict).find("NotFoundError") >= 0:
                return {"message": feature_drift_dict, "inference_name": inferencename}, 404
            elif str(feature_drift_dict).find("No data") >= 0:
                return {"message": feature_drift_dict, "inference_name": inferencename,
                        "start_time": start_time, "end_time": end_time, "data": [],
                        "prediction_count": 0}, 200
            return {"message": feature_drift_dict, "inference_name": inferencename}, 400

        return {"message": "Success get feature drift data", "inference_name": inferencename, "start_time": start_time,
                "end_time": end_time, "data": feature_drift_dict, "prediction_count": predict_count}, 200


@ns.route("/drift-monitor/prediction-over-time/<string:inferencename>")
@ns.param('end_time', 'example=2022-05-13:05', required=True)
@ns.param('start_time', 'example=2022-05-04:05', required=True)
@ns.param('model_history_id', 'Model History ID', required=True)
@ns.param('inferencename', 'Kserve Inferencename')
class PredictionOverTime(Resource):
    @ns.marshal_with(drift_metrics_output_model, code=200, skip_none=True)
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('start_time', required=True, type=str,
                            location='args', help='2022-05-04:01')
        parser.add_argument('end_time', required=True, type=str,
                            location='args', help='2022-05-13:01')
        parser.add_argument('model_history_id', required=True, type=str,
                            location='args', help='000001')

        args = parser.parse_args()

        result, monitor_setting = get_monitor_settings(inferencename)
        if result is False:
            return {"message": "Monitor is off", "inference_name": inferencename}, 400
        if 'status_code' in monitor_setting:
            return {"message": "Monitor is off", "inference_name": inferencename}, 400

        start_time, end_time, model_history_id = parsing_get_data(args)
        try:
            start_time = utils.convertTimestamp(start_time)
            end_time = utils.convertTimestamp(end_time)
        except:
            return {"message": "time parser error, time format must be yyyy-mm-dd:hh"}, 400

        result, data = get_percent_over_time(inferencename, model_history_id, start_time, end_time)
        if result is False:
            if str(data).find("No data") >= 0:
                return {"message": data, "data": [], "inference_name": inferencename}, 200
            return {"message": data, "inference_name": inferencename}, 400

        return {"message": "Success get prediction over time data", "data": data, "inference_name": inferencename}, 200


@ns.route("/drift-monitor/monitor-info/<string:inferencename>")
@ns.param('model_history_id', 'Model History ID', required=True)
@ns.param('inferencename', 'Kserve Inferencename')
class GetMonitorInfo(Resource):
    @ns.marshal_with(drift_metrics_output_model, code=200, skip_none=True)
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('model_history_id', required=True, type=str,
                            location='args', help='000001')
        args = parser.parse_args()
        model_history_id = args.get('model_history_id')
        result, inference_info = utils.LoadData('trainingdata_feature_info',
                                                f"{inferencename}_{model_history_id}").load_data(True)

        if result is False:
            return {"message": "Get Monitor Info Failed", "data": inference_info, "inference_name": inferencename}, 400

        return {"message": "Get Monitor Info Success", "data": inference_info, "inference_name": inferencename}, 200


@ns.route("/drift-monitor")
class DriftMonitorpostAPI(Resource):
    @ns.expect(base_monitor_model, validate=True)
    @ns.marshal_with(base_output_model, code=201, skip_none=True)
    def post(self):
        args = api.payload

        modelPath, trainPath, inference_name, model_id, target_label, model_type, framework, \
        drift_threshold, importance_threshold, monitor_range, low_imp_atrisk_count, \
        low_imp_failing_count, high_imp_atrisk_count, high_imp_failing_count = parsing_base_data(args)

        try:
            df = utils.read_csv(trainPath)
            if df is False:
                return {"message": "Dataset Load Error", "inference_name": inference_name}, 400

            model = utils.load_model(framework, modelPath)
            if model is False:
                return {"message": "Model Load Error", "inference_name": inference_name}, 400

            result, message = utils.validate(model, df, target_label, framework)
            if result is False:
                return {"message": message, "inference_name": inference_name}, 400

            result, message = create_info(df, inference_name, model_id, target_label, model_type)
            if result is False:
                return {"message": message, "inference_name": inference_name}, 400

            result, message = create_base(model, df, inference_name, model_id, target_label, model_type)
            if result is False:
                utils.delete_id_data("trainingdata_feature_info", f"{inference_name}_{model_id}")
                return {"message": message, "inference_name": inference_name}, 400

            result, message = add_history(inference_name, model_id)
            if result is False:
                utils.delete_id_data("trainingdata_feature_info", f"{inference_name}_{model_id}")
                utils.delete_id_data("trainingdata_base_metric", f"{inference_name}_{model_id}")
                return {"message": message, "inference_name": inference_name}, 400

            result, message = create_default_monitor_setting(inference_name, model_id, drift_threshold,
                                                             importance_threshold, monitor_range, low_imp_atrisk_count,
                                                             low_imp_failing_count, high_imp_atrisk_count,
                                                             high_imp_failing_count)
            if result is False:
                utils.delete_id_data("trainingdata_feature_info", f"{inference_name}_{model_id}")
                utils.delete_id_data("trainingdata_base_metric", f"{inference_name}_{model_id}")
                return {"message": message, "inference_name": inference_name}, 400

        except Exception as err:
            utils.delete_id_data("trainingdata_feature_info", f"{inference_name}_{model_id}")
            utils.delete_id_data("trainingdata_base_metric", f"{inference_name}_{model_id}")
            return {"message": err, "inference_name": inference_name}, 400

        logger.info(f"Drift Monitor base create Success, {inference_name}")
        return {"message": "Drift Monitor base create Success", "inference_name": inference_name}, 201


@ns.route("/drift-monitor/<string:inferencename>")
@ns.param('inferencename', 'Kserve Inferencename')
class PatchMonitorSetting(Resource):
    @ns.expect(monitor_setting_model, validate=True)
    @ns.marshal_with(base_output_model, code=200, skip_none=True)
    def patch(self, inferencename):
        args = api.payload
        if args.get('current_model') is not None:
            result, status = utils.check_index(inferencename, args.get('current_model'))
            if result is False:
                return {"message": status, "inference_name": inferencename}, 400
            if status is False:
                return {"message": "Update failed. Register the model information to update first.",
                        "inference_name": inferencename}, 400

        result, message = utils.update_data('data_drift_monitor_setting', inferencename, args)
        if result is False:
            return {"message": message, "inference_name": inferencename}, 400
        else:
            return {"message": "Monitoring settings update was successful.", "inference_name": inferencename}, 200


@ns.route("/drift-monitor/disable-monitor/<string:inferencename>")
@ns.param("inferencename", "Kserve Inferencename")
class DisableMonitor(Resource):
    @ns.marshal_with(base_output_model, code=200, skip_none=True)
    def patch(self, inferencename):
        result = disable_monitor(inferencename)
        if result is False:
            return {"message": "Disable Failed", "infernecename": inferencename}, 400

        return {"message": "Drift Monitor is disabled", "inference_name": inferencename}, 200


@ns.route("/drift-monitor/enable-monitor/<string:inferencename>")
@ns.param("inferencename", "Kserve Inferencename")
class EnableMonitor(Resource):
    @ns.marshal_with(base_output_model, code=200, skip_none=True)
    def patch(self, inferencename):
        result = enable_monitor(inferencename)
        if result is False:
            return {"message": "Enable Failed", "inferencename": inferencename}, 400

        return {"message": "Drift Monitor is enabled", "inference_name": inferencename}, 200


############################################################
# Domain Logic
############################################################

def parsing_base_data(request_data):
    model_path = request_data.get('model_path')
    train_dataset_path = request_data.get('train_dataset_path')
    inference_name = request_data.get('inference_name')
    model_id = request_data.get('model_id')
    target_label = request_data.get('target_label')
    model_type = request_data.get('model_type')
    framework = request_data.get('framework')
    drift_threshold = request_data.get('drift_threshold')
    importance_threshold = request_data.get('importance_threshold')
    monitor_range = request_data.get('monitor_range')
    low_imp_atrisk_count = request_data.get("low_imp_atrisk_count")
    low_imp_failing_count = request_data.get("low_imp_failing_count")
    high_imp_atrisk_count = request_data.get("high_imp_atrisk_count")
    high_imp_failing_count = request_data.get("high_imp_failing_count")

    local_model_path = utils.minio_client(model_path)
    local_train_path = utils.minio_client(train_dataset_path)

    return local_model_path, local_train_path, inference_name, model_id, target_label, model_type, framework, \
           drift_threshold, importance_threshold, monitor_range, low_imp_atrisk_count, \
           low_imp_failing_count, high_imp_atrisk_count, high_imp_failing_count


def parsing_get_data(request_args):
    start_time = request_args.get('start_time')
    end_time = request_args.get('end_time')
    model_history_id = request_args.get('model_history_id')

    return start_time, end_time, model_history_id


def get_monitor_settings(inference_name):
    monitor_setting_data = utils.LoadData("data_drift_monitor_setting", inference_name)
    result, monitor_setting = monitor_setting_data.load_data(True)
    return result, monitor_setting


def check_drift(drift_dict):
    high_imp_count = 0
    low_imp_count = 0
    score = drift_dict['drift score']
    feature_imp = drift_dict['feature importance']
    imp_threshold = drift_dict['threshold']['importance']
    drift_threshold = drift_dict['threshold']['drift']

    for i in feature_imp:
        if feature_imp[i] >= imp_threshold:
            if score[i] >= drift_threshold:
                high_imp_count += 1
        else:
            if score[i] >= drift_threshold:
                low_imp_count += 1

    return high_imp_count, low_imp_count


def check_status(high_imp_count, low_imp_count, low_imp_atrisk_count, low_imp_failing_count, high_imp_atrisk_count,
                 high_imp_failing_count):
    if high_imp_failing_count != 0:
        if high_imp_failing_count <= high_imp_count:
            return 'failing'
    if low_imp_failing_count != 0:
        if low_imp_failing_count <= low_imp_count:
            return 'failing'
    if high_imp_atrisk_count != 0:
        if high_imp_atrisk_count <= high_imp_count:
            return 'atrisk'
    if low_imp_atrisk_count != 0:
        if low_imp_atrisk_count <= low_imp_count:
            return 'atrisk'
    return 'pass'


def driftMonitor():
    while 1:
        query = {
            "match": {
                "status": "enable"
            }
        }
        result, items = utils.search_scroll_index('data_drift_monitor_setting', query=query)
        if result is False:
            logger.warning(items)
        else:
            producer = makeproducer()
            if producer is False:
                logger.warning('kafka.errors.NoBrokersAvailable: NoBrokersAvailable')
            else:
                for item in items:
                    inference_name = item['_id']
                    model_id = item['_source']['current_model']
                    drift_threshold = item['_source']['drift_threshold']
                    importance_threshold = item['_source']['importance_threshold']
                    monitor_range = item['_source']['monitor_range']
                    low_imp_atrisk_count = item['_source']['low_imp_atrisk_count']
                    low_imp_failing_count = item['_source']['low_imp_failing_count']
                    high_imp_atrisk_count = item['_source']['high_imp_atrisk_count']
                    high_imp_failing_count = item['_source']['high_imp_failing_count']
                    start_time = f"now-{monitor_range}"
                    end_time = "now"

                    result, feature_drift_dict, predict_count = get_feature_drift_dict(inference_name, model_id,
                                                                                       start_time,
                                                                                       end_time, drift_threshold,
                                                                                       importance_threshold)
                    if result is False or predict_count < 100:
                        drift_result = 'unknown'
                    else:
                        high_imp_count, low_imp_count = check_drift(feature_drift_dict)
                        drift_result = check_status(high_imp_count, low_imp_count, low_imp_atrisk_count,
                                                    low_imp_failing_count,
                                                    high_imp_atrisk_count, high_imp_failing_count)
                    try:
                        produceKafka(producer, {"inference_name": inference_name, "result": drift_result})
                    except:
                        logger.warning("producer is none")
                try:
                    producer.close()
                except:
                    logger.warning("producer error")
        time.sleep(30)


def disable_monitor(inference_name):
    result, message = utils.update_data('data_drift_monitor_setting', inference_name, {"status": "disable"})
    if result is False:
        return False
    return True


def enable_monitor(inference_name):
    result, message = utils.update_data('data_drift_monitor_setting', inference_name, {"status": "enable"})
    if result is False:
        return False
    return True


############################################################
# Main
############################################################


if __name__ == '__main__':
    Process(target=driftMonitor).start()
    consumeKafka()

    # test
    # app.run(threaded=True)
