import re
from datetime import datetime


class StockCodeConverter:
    @staticmethod
    def to_sina_date_str(ts_date_str: str) -> str | None:
        if not ts_date_str:
            return None
        ts_format = "%Y%m%d"
        sina_format = "%Y-%m-%d"
        date_obj = datetime.strptime(ts_date_str, ts_format)
        return datetime.strftime(date_obj, sina_format)

    @staticmethod
    def sina_to_tushare(sina_code: str) -> str | None:
        if not sina_code:
            return None
        if "sz" in sina_code:
            return sina_code.replace("sz", "") + ".SZ"
        elif "sh" in sina_code:
            return sina_code.replace("sh", "") + ".SH"
        else:
            return None

    @staticmethod
    def tushare_to_sina(tushare_code: str) -> str | None:
        if not tushare_code:
            return None
        tmp = tushare_code.split(".")
        return tmp[1].lower() + tmp[0]

    @staticmethod
    def to_symbol_hash(tushare_code: str) -> int | None:
        if not tushare_code:
            return None
        code = tushare_code.split(".")[0]
        code = ''.join(re.findall(r'\d', code))
        return int(code)

    @staticmethod
    def sina_to_symbol(sina_code: str) -> int | None:
        if not sina_code:
            return None
        return int(sina_code.replace("sz", "").replace("sh", "").replace("bj", "").replace("ti", "").replace("r", "").replace("R", ""))

    @staticmethod
    def tushare_to_symbol(tushare_code: str) -> int | None:
        return StockCodeConverter.sina_to_symbol(StockCodeConverter.tushare_to_sina(tushare_code))
