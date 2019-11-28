# from celery.schedules import crontab
# from datetime import timedelta
from kombu import Queue
from kombu import Exchange
import sys
sys.path.append('..')
import configparser

cf = configparser.ConfigParser()
cf.read("config.conf")
broker_ip = cf.get("celery", "broker_ip")
broker_port = cf.get("celery", "broker_port")
broker_db = cf.get("celery", "broker_db")
# broker_user = cf.get("celery", "broker_user")
# broker_password = cf.get("celery", "broker_password")
backend_ip = cf.get("celery", "backend_ip")
backend_port = cf.get("celery", "backend_port")
backend_db = cf.get("celery", "backend_db")
# backend_user = cf.get("celery", "backend_user")
# backend_password = cf.get("celery", "backend_password")
worker_concurrency = cf.getint("celery", "worker_concurrency_gpu")
broker_pwd = cf.get("celery", "broker_pwd")
backend_pwd = cf.get("celery", "backend_pwd")
queue_name = cf.get("celery", "queue_name")
routing_key_name = cf.get("celery", "routing_key_name")
result_serializer = 'msgpack'
task_serializer = 'msgpack'
accept_content = ['json', 'msgpack']


# broker_url = 'amqp://%s:%s@%s:%s/vhost' % (broker_user, broker_password,broker_ip, broker_port)
broker_url = "redis://:%s@%s:%s/%s" % (broker_pwd, broker_ip, broker_port, broker_db)
# result_backend = 'amqp://%s:%s@%s:%s/vhost' % (backend_user, backend_password, backend_ip, backend_port)
result_backend = "redis://:%s@%s:%s/%s" % (backend_pwd, backend_ip, backend_port, backend_db)
# timezone = "Asia/Shangh"
# worker_concurrency = 8
worker_prefetch_multiplier = 1
# result_exchange = 'agent'
result_exchange_type = 'direct'
result_expires = 5
# worker_max_tasks_per_child = 10000
task_time_limit = 30
C_FORCE_ROOT = True
# task_annotations = {'celery_app_tmp.task0.faceRec': {'rate_limit': '50/s'},
#                     'celery_app_tmp.task1.faceRec': {'rate_limit': '50/s'},
#                     'celery_app_tmp.task0.featureExtract': {'rate_limit': '50/s'},
#                     'celery_app_tmp.task1.featureExtract': {'rate_limit': '50/s'},}
# broker_pool_limit = 100
# redis_socket_connect_timeout = 5
# redis_socket_timeout = 5
# redis_max_connections = 32
task_queues = (
    # Queue('default', exchange=Exchange('default'), routing_key='default'),
    Queue(queue_name, exchange=Exchange(queue_name), routing_key=routing_key_name),
    #Queue('gpu_1', exchange=Exchange('gpu_1'), routing_key='gpu_1'),
    # Queue('download', exchange=Exchange('download'), routing_key='download'),
)

task_routes = {
                'celery_app.task.headsDet': {'queue': queue_name, 'routing_key': routing_key_name},
}

