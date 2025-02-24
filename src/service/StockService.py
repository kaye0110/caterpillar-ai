import logging

import pandas as pd

from src.config.DBWrapper import DBWrapper
from src.infra.converter.stock_code_converter import StockCodeConverter


class StockService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_tx_index_by_code(self, codes: [str]) -> pd.DataFrame | None:
        if codes is None:
            return pd.DataFrame()
        stock_codes = ",".join(f"'{item}'" for item in codes)
        query = f"""
                select l3_code as index_code3,
                l3_name as index_name3,
                l2_code as index_code2,
                l2_name as index_name2,
                l1_code as index_code1,
                l1_name as index_name1,
                ts_code as stock_code,
                name as stock_name,
                'normal' as label_source
                 from ts_index_member tim  
                where tim.ts_code in ({stock_codes}) ;
                """
        with DBWrapper() as db:
            results = db.execute_query(query)
            if results is not None:
                return pd.DataFrame(results)
        self.logger.info(f"no label found for codes:{stock_codes}")

        return pd.DataFrame()

    def get_ths_index_by_code(self, codes: [str]) -> pd.DataFrame | None:
        if codes is None:
            return pd.DataFrame()
        stock_codes = ",".join(f"'{item}'" for item in codes)

        query = f"""
                select idx.name as index_name3, 
                idx.ts_code as index_code3,
                '' as index_name2,
                '' as index_code2,
                '' as index_name1,
                '' as index_code1, 
                tm.code as stock_code ,
                tm.name as stock_name ,
                'ths' as label_source
                from ths_member tm , ths_index idx where idx.ts_code  = tm.ts_code 
                and tm.code in ({stock_codes}) ; 
                """
        with DBWrapper() as db:
            results = db.execute_query(query)
            if results is not None:
                return pd.DataFrame(results)
        self.logger.info(f"no label found for codes:{stock_codes}")

        return pd.DataFrame()

    def get_last_trade_date_from_index_const(self, index_code) -> str | None:
        query = f"""
                select max(trade_date) as trade_date from ts_index_constituents where index_code = '{index_code}' ;
                """
        with DBWrapper() as db:
            results = db.execute_query(query)
            if results is None:
                self.logger.info(f"no trade date found for index:{index_code}")
                return None

            return results[0]['trade_date']

    def get_ts_index_code_by_name(self, label: str) -> str | None:
        query = """
                        select ts_code from ts_index_info where fullname like %s
                        """
        label = f"%{label}%"
        with DBWrapper() as db:
            results = db.execute_query(query, (label,))
            if results is None:
                self.logger.info(f"no label found for codes:{label}")
                return None
            df = pd.DataFrame(results)
            return df['ts_code'].tolist()

    def get_stock_by_index_code(self, index_code: str) -> pd.DataFrame | None:
        if index_code is None:
            return None

        trade_date_str = self.get_last_trade_date_from_index_const(index_code)
        if trade_date_str is None:
            return None

        query = """
                        select '' as index_name3, 
                            const.index_code as index_code3,
                            '' as index_name2,
                            '' as index_code2,
                            '' as index_name1,
                            '' as index_code1, 
                            const.con_code as stock_code ,
                            '' as stock_name ,
                            const.weight as weight
                            from ts_index_constituents const
                            where const.index_code = %s 
                            order by const.weight desc
                    """
        with DBWrapper() as db:
            results = db.execute_query(query, (index_code,))
            if results is not None:
                return pd.DataFrame(results)

    def get_stock_code_by_index_const(self, labels: [str]) -> pd.DataFrame | None:

        df_array = []
        for label in labels:
            index_code_array = self.get_ts_index_code_by_name(label)
            if index_code_array is None or len(index_code_array) == 0:
                continue
            for index_code in index_code_array:
                df = self.get_stock_by_index_code(index_code)
                if df is None:
                    continue
                df_array.append(df)

        if len(df_array) > 0:
            return pd.concat(df_array)

    def get_stock_code_by_index_member(self, labels: [str]) -> pd.DataFrame | None:
        query = """
                select l3_code as index_code3,
                l3_name as index_name3,
                l2_code as index_code2,
                l2_name as index_name2,
                l1_code as index_code1,
                l1_name as index_name1,
                ts_code as stock_code,
                name as stock_name,
                'normal' as label_source
                from ts_index_member tim  
                where ( tim.l1_name like %s or  tim.l2_name like %s or  tim.l3_name like %s )
                """
        with DBWrapper() as db:
            tmp = []
            for label in labels:
                query_param = f'%{label}%'
                results = db.execute_query(query, (query_param, query_param, query_param,))
                if results is not None:
                    tmp.append(pd.DataFrame(results))

            if len(tmp) > 0:
                return pd.concat(tmp)
        self.logger.info(f"no stock code found for labels:{labels}")

        return pd.DataFrame()

    def get_stock_code_by_ths_labels(self, labels: [str]) -> pd.DataFrame | None:
        query = """
                select idx.name as index_name3, 
                idx.ts_code as index_code3,
                '' as index_name2,
                '' as index_code2,
                '' as index_name1,
                '' as index_code1, 
                tm.code as stock_code ,
                tm.name as stock_name ,
                'ths' as label_source
                from ths_member tm , ths_index idx 
                where idx.ts_code  = tm.ts_code 
                and idx.exchange = 'A'
                and idx.name like %s 
                """
        with DBWrapper() as db:
            tmp = []
            for label in labels:
                query_param = f'%{label}%'
                results = db.execute_query(query, (query_param,))
                if results is not None:
                    tmp.append(pd.DataFrame(results))

            if len(tmp) > 0:
                return pd.concat(tmp)
        self.logger.info(f"no stock code found for labels:{labels}")

        return pd.DataFrame()

    def get_ths_labels(self):
        query = """
                select i.ts_code as index_code, i.name as index_name, i.exchange as index_change , i.type as index_type, i.list_date  as index_list_date, m.code as stock_code, m.name as stock_name 
                from ths_member m, ths_index i where i.ts_code = m.ts_code order by m.ts_code asc, i.ts_code asc
                """
        with DBWrapper() as db:
            results = db.execute_query(query)
            if results is not None:
                # self.logger.info(f"Retrieved stock price data for {ts_code} from {start_date} to {end_date}.")
                return pd.DataFrame(results)
            else:
                self.logger.error("Failed to retrieve stock price data.")
                return None

    def get_all_stocks(self):
        """Retrieve all stocks from ts_stock_basic."""
        query = "SELECT * FROM ts_stock_basic WHERE list_status = 'L' "
        with DBWrapper() as db:
            results = db.execute_query(query)
            if results is not None:
                # self.logger.info("Retrieved all stocks successfully.")
                return pd.DataFrame(results)
            else:
                self.logger.error("Failed to retrieve stocks.")
                return None

    def get_stock_price_data(self, ts_code, start_date, end_date):
        """Retrieve stock price data for a given ts_code and date range."""
        query = """
        SELECT * FROM ts_stock_price_daily
        WHERE symbol_hash = %s AND ts_code = %s AND trade_date BETWEEN %s AND %s ORDER BY trade_date
        """
        with DBWrapper() as db:
            results = db.execute_query(query, (StockCodeConverter.tushare_to_symbol(ts_code), ts_code, start_date, end_date))
            if results is not None:
                # self.logger.info(f"Retrieved stock price data for {ts_code} from {start_date} to {end_date}.")
                return pd.DataFrame(results)
            else:
                self.logger.error("Failed to retrieve stock price data.")
                return None

    def get_stock_price_adj_data(self, ts_code, start_date, end_date):
        """Retrieve adjusted stock price data for a given ts_code."""
        query = ("SELECT * FROM ts_stock_price_adj WHERE symbol_hash = %s AND ts_code = %s "
                 " AND trade_date BETWEEN %s AND %s "
                 " AND adj_type = 'hfq' AND adj_freq = 'D' ORDER BY trade_date")
        with DBWrapper() as db:
            results = db.execute_query(query, (StockCodeConverter.tushare_to_symbol(ts_code), ts_code, start_date, end_date))
            if results is not None:
                # self.logger.info(f"Retrieved adjusted stock price data for {ts_code}.")
                return pd.DataFrame(results)
            else:
                self.logger.error("Failed to retrieve adjusted stock price data.")
                return None

    def merge_price_and_adj_data(self, ts_code, start_date, end_date):
        try:
            """Merge daily price data and adjusted price data based on ts_code and trade_date."""
            price_data = self.get_stock_price_data(ts_code, start_date, end_date)
            if price_data is None or len(price_data) == 0:
                return None

            adj_data = self.get_stock_price_adj_data(ts_code, start_date, end_date)
            if adj_data is None or len(adj_data) == 0:
                return None

            if abs(len(adj_data) - len(price_data)) != 0:
                diff = price_data[~price_data['trade_date'].isin(adj_data['trade_date'])]
                trade_date_array = diff['trade_date'].tolist()
                self.logger.error(f"Number of rows in price data and adjusted price data do not match. {ts_code} miss:{trade_date_array}")

                return None

            # Merge with suffixes to distinguish columns
            merged_data = pd.merge(
                price_data, adj_data,
                on=['ts_code', 'symbol_hash', 'trade_date'],
                how='left',
                suffixes=('', '_adj')
            )

            if abs(len(adj_data) - len(merged_data)) > 10:
                return None

            # Drop the '_adj' suffixed columns if they exist in daily data
            for column in adj_data.columns:
                if column in price_data.columns and f"{column}_adj" in merged_data.columns:
                    merged_data.drop(f"{column}_adj", axis=1, inplace=True)

            assert abs(len(adj_data) - len(merged_data)) < 10, "The number of rows in adj_data and merged_data should be the same."
            assert abs(len(price_data) - len(merged_data)) < 10, "The number of rows in price_data and merged_data should be the same."

            return merged_data


        except Exception as e:
            self.logger.error(f"Error occurred while merging price and adj data: {e}")

    def get_stock_by_code(self, stock_code: str):
        """Retrieve all stocks from ts_stock_basic."""
        query = "SELECT t.ts_code as stock_code, t.* FROM ts_stock_basic t WHERE list_status = 'L' and ts_code = %s"
        with DBWrapper() as db:
            results = db.execute_query(query, (stock_code,))
            if results is not None:
                # self.logger.info("Retrieved all stocks successfully.")
                return pd.DataFrame(results)
            else:
                self.logger.error("Failed to retrieve stocks.")
                return None
