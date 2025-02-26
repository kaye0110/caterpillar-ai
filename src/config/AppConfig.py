import logging
import os

import mysql.connector
import redis
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables from .env file
load_dotenv()
import sys
sys.stdout.reconfigure(encoding='utf-8')  # 如果你的Python版本支持的话

class AppConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # MySQL configuration
        self.mysql_config = {
            'host': os.getenv('MYSQL_HOST'),
            'port': int(os.getenv('MYSQL_PORT')),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'database': os.getenv('MYSQL_DATABASE')
        }
        self.mysql_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="mysql_pool",
            pool_size=5,
            **self.mysql_config
        )

        # Redis configuration
        self.redis_config = {
            'host': os.getenv('REDIS_HOST'),
            'port': int(os.getenv('REDIS_PORT')),
            'db': int(os.getenv('REDIS_DB'))
        }
        self.redis_pool = redis.ConnectionPool(**self.redis_config)

        # Elasticsearch configuration
        self.es_config = {
            'hosts': [f"http://{os.getenv('ELASTICSEARCH_HOST')}:{os.getenv('ELASTICSEARCH_PORT')}"],
            'basic_auth': (os.getenv('ELASTICSEARCH_USER'), os.getenv('ELASTICSEARCH_PASSWORD'))
        }
        self.es_client = Elasticsearch(**self.es_config)

        self.caterpillar_web_config = {
            'endpoint': os.getenv("CATERPILLAR_ENDPOINT"),
            'connect_timeout': 10,  # 连接超时 10 秒
            'read_timeout': 5 * 3600  # 读取超时 5 小时
        }

        # 创建一个日志格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s:%(filename)s:%(lineno)d - %(levelname)s - %(message)s')
        # 创建一个控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 创建一个文件处理器
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        output_path = os.path.join(project_root, 'logs', 'app.log')

        file_handler = logging.FileHandler(output_path)
        file_handler.setFormatter(formatter)

        # 获取根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # 将处理器添加到根日志记录器
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        self.logger = logging.getLogger(__name__)

    def get_mysql_connection(self):
        return self.mysql_pool.get_connection()

    def get_redis_connection(self):
        return redis.Redis(connection_pool=self.redis_pool)

    def get_es_connection(self):
        return self.es_client

    def validate_mysql_connection(self):
        try:
            conn = self.get_mysql_connection()
            conn.ping(reconnect=True)
            conn.close()
            self.logger.info("MySQL connection is valid.")
            return True
        except Exception as e:
            self.logger.error(f"MySQL connection failed: {e}")
            return False

    def validate_redis_connection(self):
        try:
            redis_conn = self.get_redis_connection()
            redis_conn.ping()
            self.logger.info("Redis connection is valid.")
            return True
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            return False

    def validate_es_connection(self):
        try:
            if self.es_client.ping():
                self.logger.info("Elasticsearch connection is valid.")
                return True
            else:
                self.logger.error("Elasticsearch connection failed.")
                return False
        except Exception as e:
            self.logger.error(f"Elasticsearch connection failed: {e}")
            return False
