import logging

import numpy as np
import pandas as pd

from src.config.DBWrapper import DBWrapper


class ChooseStockService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_last_profit(self):
        query = """
        SELECT 
            *
        FROM 
            stock_profit sp
        JOIN 
            (SELECT ts_code, MAX(end_date) as max_end_date FROM stock_profit GROUP BY ts_code) sp_max
        ON 
            sp.ts_code = sp_max.ts_code AND sp.end_date = sp_max.max_end_date
        """

        with DBWrapper() as db:
            results = db.execute_query(query)
            return pd.DataFrame(results)

    def get_last_balance_sheet(self):
        query = """
        SELECT 
            *
        FROM 
            stock_balance_sheet sb
        JOIN 
            (SELECT ts_code, MAX(end_date) as max_end_date FROM stock_balance_sheet GROUP BY ts_code) sb_max
        ON 
            sb.ts_code = sb_max.ts_code AND sb.end_date = sb_max.max_end_date
        """

        with DBWrapper() as db:
            results = db.execute_query(query)
            return pd.DataFrame(results)

    def get_last_cash_flow(self):
        query = """
        SELECT 
            *
        FROM 
            stock_cash_flow scf
        JOIN 
            (SELECT ts_code, MAX(end_date) as max_end_date FROM stock_cash_flow GROUP BY ts_code) scf_max
        ON 
            scf.ts_code = scf_max.ts_code AND scf.end_date = scf_max.max_end_date
        """

        with DBWrapper() as db:
            results = db.execute_query(query)
            return pd.DataFrame(results)

    def choose_stock(self):
        # 读取数据到DataFrame
        df_profit = self.get_last_profit()
        df_balance_sheet = self.get_last_balance_sheet()
        df_cash_flow = self.get_last_cash_flow()

        # 合并数据
        df = df_profit.merge(df_balance_sheet, on='ts_code').merge(df_cash_flow, on='ts_code')

        # 计算一些关键的财务比率
        df['roe'] = df['n_income'] / df['total_hldr_eqy_exc_min_int'].replace(0, np.nan)  # 净资产收益率
        df['roa'] = df['n_income'] / df['total_assets'].replace(0, np.nan)  # 总资产收益率
        df['debt_to_equity'] = df['total_liab'] / df['total_hldr_eqy_exc_min_int'].replace(0, np.nan)  # 负债权益比
        df['current_ratio'] = df['total_cur_assets'] / df['total_cur_liab'].replace(0, np.nan)  # 流动比率
        df['pe_ratio'] = df['total_assets'] / df['n_income'].replace(0, np.nan)  # 市盈率

        # 处理异常值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 多因子评分
        df['value_score'] = 1 / df['pe_ratio'].replace(0, np.nan)  # 价值因子

        # 计算同比增长率
        df['growth_score'] = df['total_revenue'].pct_change(periods=4).replace([np.inf, -np.inf], np.nan)  # 成长因子

        df['quality_score'] = df['roe']  # 质量因子

        # 标准化处理
        df['value_score'] = (df['value_score'] - df['value_score'].mean()) / df['value_score'].std()
        df['growth_score'] = (df['growth_score'] - df['growth_score'].mean()) / df['growth_score'].std()
        df['quality_score'] = (df['quality_score'] - df['quality_score'].mean()) / df['quality_score'].std()

        # 综合评分
        df['total_score'] = df[['value_score', 'growth_score', 'quality_score']].sum(axis=1)

        # 选择有潜力的股票
        df = df.sort_values(by='total_score', ascending=False)

        # 选择前20个股票
        top_20_stocks = df.head(20)

        print(top_20_stocks[['ts_code', 'total_score', 'value_score', 'growth_score', 'quality_score']])


if __name__ == '__main__':
    ChooseStockService().choose_stock()
