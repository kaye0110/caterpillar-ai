import json
import logging

import requests

from src.config.AppConfig import AppConfig


class StockPriceProvider:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.caterpillar_web_config = AppConfig().caterpillar_web_config
        self.caterpillar_endpoint = self.caterpillar_web_config.get('endpoint', "http://127.0.0.1:8800")
        self.connect_timeout = self.caterpillar_web_config.get('connect_timeout', 10)
        self.read_timeout = self.caterpillar_web_config.get('read_timeout', 5 * 3600)

    def calculate_indicator(self, batch_code: str, start_date: str, end_date: str, stock_codes: [str]) -> (str, str):
        url = f"{self.caterpillar_endpoint}/price/indicator/calculate"

        payload = {
            "batch_code": batch_code,
            "start_date": start_date,
            "end_date": end_date,
            "ts_codes": stock_codes
        }
        try:
            # 发送 POST 请求
            response = requests.post(url, json=payload, timeout=(self.connect_timeout, self.read_timeout))
            # 检查响应状态
            if response.status_code == 200:
                # 解析 JSON 响应
                result = response.json()

                if result.get("status") == "200":
                    # 提取关键数据
                    file_path = result["data"]["file_path"]
                    file_name = result["data"]["file_name"]

                    return file_path, file_name
                else:
                    self.logger.error(f"接口返回错误: {result.get('message')}")
            else:
                self.logger.error(f"请求失败，状态码: {response.status_code}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"网络请求异常: {str(e)}")
        except json.JSONDecodeError:
            self.logger.error("响应不是有效的 JSON 格式")

        return None, None

    def download_indicator(self, export_path: str, batch_code: str, start_date: str, end_date: str, stock_codes: [str]) -> bool:
        url = f"{self.caterpillar_endpoint}/price/indicator/download"

        payload = {
            "batch_code": batch_code,
            "start_date": start_date,
            "end_date": end_date,
            "ts_codes": stock_codes
        }
        try:
            # 发送 POST 请求
            response = requests.post(url, json=payload, timeout=(self.connect_timeout, self.read_timeout), stream=True)
            # 检查响应状态
            if response.status_code == 200:
                # 打开一个文件以二进制写模式保存响应内容
                with open(export_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # 过滤掉保持活动连接的空块
                            file.write(chunk)

                return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"网络请求异常: {str(e)}")
        except json.JSONDecodeError:
            self.logger.error("响应不是有效的 JSON 格式")

        return False
