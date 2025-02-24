import logging

import pandas as pd
import talib

from src.tools.pycaterpillar_wrapper import time_counter


class IndicatorService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_drop_columns = ["gmt_created", "gmt_modified", "label_json", "symbol_hash", "creator", "modifier", "is_deleted", "status", "tenant"
            , "adj_type", "adj_freq", "adj_factor", "tenant", "tenant", "tenant"]
        self.price_change_columns = ["vol", "amount", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount", "buy_md_vol", "buy_md_amount",
                                     "sell_md_vol", "sell_md_amount", "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
                                     "sell_elg_vol", "sell_elg_amount", "net_mf_vol", "net_mf_amount", "sell_lg_vol", "sell_lg_amount", "buy_elg_vol", "buy_elg_amount",
                                     "turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
                                     "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share", "total_mv", "circ_mv"]
        self.periods = {
            "short5": 5, "medium7": 7, "medium12": 12, "long14": 14, "long19": 19
        }

        self.rsi_period = {"short": {"slow": 9, "fast": 14}, "medium": {"slow": 14, "fast": 21}}
        self.macd_period = {"short": [6, 13, 5], "medium": [12, 26, 9]}
        self.kdj_period = {"short": [9, 3, 3], "medium": [14, 3, 3], "medium2": [14, 5, 3]}
        self.bbands_period = {"short": [15, 1.5, 2], "medium": [20, 2, 2], "long": [40, 2.5, 2.5]}
        self.ultosc_period = {"short": [3, 6, 9], "medium": [7, 14, 28], "medium2": [6, 12, 24]}

    def prepare_price(self, price_data: pd.DataFrame) -> pd.DataFrame:
        price_data.drop(columns=self.price_drop_columns, inplace=True)
        price_data.rename(columns={col: f'indicator_{col}' for col in self.price_change_columns}, inplace=True)

        return price_data

    @staticmethod
    # talib 没有直接提供 SMMA 的计算方法，但可以通过自定义函数实现
    def smma(series, period):
        return series.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def calculate_macd_cross(macd, macdsignal):
        # 初始化金叉和死叉信号的 Series
        gold_cross = pd.Series(index=macd.index, data=0)
        death_cross = pd.Series(index=macd.index, data=0)

        # 计算金叉和死叉信号
        for i in range(1, len(macd)):
            if macd[i] > macdsignal[i] and macd[i - 1] <= macdsignal[i - 1]:
                # 金叉信号
                gold_cross[i] = 1
            elif macd[i] < macdsignal[i] and macd[i - 1] >= macdsignal[i - 1]:
                # 死叉信号
                death_cross[i] = 1

        return gold_cross, death_cross

    @staticmethod
    def calculate_stochf_crosses(stochf_k, stochf_d):
        # 确保输入是 Pandas Series
        stochf_k = pd.Series(stochf_k)
        stochf_d = pd.Series(stochf_d)

        # 计算黄金交叉：%K 从下向上穿过 %D
        gold_cross = (stochf_k.shift(1) < stochf_d.shift(1)) & (stochf_k > stochf_d)

        # 计算死亡交叉：%K 从上向下穿过 %D
        death_cross = (stochf_k.shift(1) > stochf_d.shift(1)) & (stochf_k < stochf_d)

        return gold_cross.fillna(0), death_cross.fillna(0)

    @time_counter(logger_name=__name__)
    def generate_talib(self, df: pd.DataFrame) -> pd.DataFrame:
        df['indicator_talib_AD'] = talib.AD(df['high'], df['low'], df['close'], df['indicator_vol'])
        df['indicator_talib_ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['indicator_vol'], fastperiod=3, slowperiod=10)

        df['indicator_total_mv_group'] = pd.qcut(df['indicator_total_mv'].fillna(0).astype(float), 5, labels=False) + 1

        for period in self.periods:
            df[f'indicator_talib_ADX_{period}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.periods[period])
            df[f'indicator_talib_ADXR_{period}'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=self.periods[period])

            aroondown, aroonup = talib.AROON(df['high'], df['low'], timeperiod=self.periods[period])
            df[f'indicator_talib_aroondown_{period}'] = aroondown
            df[f'indicator_talib_aroonup_{period}'] = aroonup

            df[f'indicator_talib_aroonosc_{period}'] = talib.AROONOSC(df['high'], df['low'], timeperiod=self.periods[period])
            df[f'indicator_talib_ATR_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.periods[period])
            # CCI 是一种动量指标，主要用于识别新的趋势或警告极端市场条件。它通过比较当前价格与过去一段时间的平均价格来衡量价格的波动性。
            df[f'indicator_talib_CCI_{period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=self.periods[period])

            # CMO 是一种动量指标，用于衡量价格的动量强度。它的计算基于一定周期内价格上涨和下跌的差值。
            df[f'indicator_talib_CMO_{period}'] = talib.CMO(df['close'], timeperiod=self.periods[period])
            # 计算两个时间序列之间的皮尔逊相关系数
            df[f'indicator_talib_CORREL_buy_elg_vol_close_{period}'] = talib.CORREL(df['indicator_buy_elg_vol'], df['close'], timeperiod=self.periods[period])
            df[f'indicator_talib_CORREL_sell_elg_vol_close_{period}'] = talib.CORREL(df['indicator_sell_elg_vol'], df['close'], timeperiod=self.periods[period])

            # Directional Movement Index（DX）是技术分析中用于衡量价格趋势强度的指标。它是通过计算正向和负向的动向指标（+DI和-DI）来得出的。DX的值越高，表示趋势越强。
            df[f'indicator_talib_DX_{period}'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=self.periods[period])

            # KAMA 通过考虑市场的波动性来调整其平滑系数，从而在趋势市场中更快地响应价格变化，而在震荡市场中则更平滑。
            kama = talib.KAMA(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(kama, name=f'indicator_talib_KAMA_{period}')], axis=1)

            indicator = talib.LINEARREG(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_LINEARREG_{period}')], axis=1)

            indicator = talib.LINEARREG_ANGLE(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_LINEARREG_ANGLE_{period}')], axis=1)

            indicator = talib.LINEARREG_INTERCEPT(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_LINEARREG_INTERCEPT_{period}')], axis=1)

            indicator = talib.LINEARREG_SLOPE(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_LINEARREG_SLOPE_{period}')], axis=1)

            # **简单移动平均线（SMA）**：通过计算一定时间周期内价格的算术平均值来平滑数据。
            indicator = talib.SMA(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_SMA_{period}')], axis=1)

            # **指数移动平均线（EMA）**：对最近的价格数据赋予更高的权重，更敏感于价格变化。
            indicator = talib.EMA(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_EMA_{period}')], axis=1)

            # **加权移动平均线（WMA）**：对不同时间点的价格赋予不同的权重。
            indicator = talib.WMA(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_WMA_{period}')], axis=1)

            # **三角移动平均线（TMA）**：通过对简单移动平均线再次平滑来减少波动。
            indicator = talib.TRIMA(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_TRIMA_{period}')], axis=1)

            # ** 双指数移动平均线（DEMA） ** 和 ** 三指数移动平均线（TEMA） ** ：通过多次指数平滑来减少滞后。
            # DEMA（Double Exponential Moving Average，双指数移动平均线）是一种技术分析指标，用于平滑价格数据并识别价格趋势。与传统的简单移动平均线（SMA）和指数移动平均线（EMA）相比，DEMA通过减少滞后性提供了更快速的响应。
            indicator = talib.DEMA(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_DEMA_{period}')], axis=1)

            indicator = talib.TEMA(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_TEMA_{period}')], axis=1)

            # **平滑移动平均线（SMMA）**：类似于 EMA，但平滑效果更强。
            indicator = IndicatorService.smma(df['close'], period=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_smma_{period}')], axis=1)

            macd, macdsignal, macdhist = talib.MACDFIX(df['close'], signalperiod=self.periods[period])
            macd.fillna(0, inplace=True)
            macdsignal.fillna(0, inplace=True)
            macdhist.fillna(0, inplace=True)

            gold_cross, death_cross = IndicatorService.calculate_macd_cross(macd, macdsignal)

            df = pd.concat([df, pd.Series(macd, name=f'indicator_talib_macd_fix_{period}')], axis=1)
            df = pd.concat([df, pd.Series(macdsignal, name=f'indicator_talib_macdsignal_fix_{period}')], axis=1)
            df = pd.concat([df, pd.Series(macdhist, name=f'indicator_talib_macdhist_fix_{period}')], axis=1)
            df = pd.concat([df, pd.Series(gold_cross, name=f'indicator_talib_macd_fix_gold_cross_{period}')], axis=1)
            df = pd.concat([df, pd.Series(death_cross, name=f'indicator_talib_macd_fix_death_cross_{period}')], axis=1)

            # Money Flow Index（MFI）是一个技术分析指标，用于衡量资金流入和流出某一资产的强度。它结合了价格和交易量的数据，通常用于识别超买或超卖的市场条件。
            # MFI 的取值范围在 0 到 100 之间，通常认为高于 80 表示超买，低于 20 表示超卖。
            indicator = talib.MFI(df['high'], df['low'], df['close'], df['indicator_vol'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_MFI_{period}')], axis=1)

            # 中点是指在特定时间段内，最高价和最低价的平均值。这个指标可以帮助交易者识别价格的中间趋势，提供对市场价格波动的更好理解。
            indicator = talib.MIDPOINT(df['high'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_midpoint_{period}')], axis=1)

            # MIDPRICE 是技术分析中的一个指标，用于计算一段时间内最高价和最低价的中点价格。它可以帮助交易者识别市场的中间趋势，尤其是在价格波动较大的市场中。
            indicator = talib.MIDPRICE(df['high'], df['low'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_midprice_{period}')], axis=1)

            # MIN 函数用于计算在指定周期内的最低值。这个指标可以帮助投资者识别某一段时间内的最低价格，从而判断市场的支撑位或潜在的买入点。
            indicator = talib.MIN(df['low'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_MIN_{period}')], axis=1)

            indicator = talib.MAX(df['high'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_MAX_{period}')], axis=1)

            # 它表示市场的下跌动能。数值越高，表示下跌动能越强。
            indicator = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_MINUS_DI_{period}')], axis=1)

            # 如果 MINUS_DM > PLUS_DM，市场可能处于下跌趋势。
            # 如果 MINUS_DM < PLUS_DM，市场可能处于上涨趋势。
            # DMI（Directional Movement Index）：由 PLUS_DM 和 MINUS_DM 构成，用于判断趋势方向。
            # ADX（Average Directional Index）：基于 PLUS_DM 和 MINUS_DM 计算，用于衡量趋势的强度。
            indicator = talib.MINUS_DM(df['high'], df['low'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_MINUS_DM_{period}')], axis=1)

            # MOM（Momentum）指标是一个动量指标，用于衡量价格在特定时间段内的变化。它通过计算当前价格与过去某一价格之间的差异来反映价格的动量。
            # MOM指标可以帮助投资者识别价格趋势的强弱和可能的反转点。
            # 当MOM指标从负转正时，可能预示着价格的上升趋势；而当MOM指标从正转负时，可能预示着价格的下降趋势。
            indicator = talib.MOM(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_MOM_{period}')], axis=1)

            # - PLUS_DI 指标的值越高，表示上升动能越强，市场可能处于上升趋势。
            # - 如果 PLUS_DI 高于某个阈值（例如 20），可能意味着市场有较强的上升趋势。
            # - 结合其他指标（如 MINUS_DI）可以更好地判断市场趋势。
            indicator = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_PLUS_DI_{period}')], axis=1)

            # `PLUS_DM`（Plus Directional Movement）是技术分析中用于衡量价格上升动能的指标。
            # 它是方向性运动指标（DMI）的一部分，通常与负方向性运动（MINUS_DM）和平均方向性指数（ADX）一起使用。
            # `PLUS_DM`的计算方法是通过比较当前周期的最高价与前一个周期的最高价来确定的，如果当前周期的最高价高于前一个周期的最高价，则认为存在正向动能。
            indicator = talib.PLUS_DM(df['high'], df['low'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_PLUS_DM_{period}')], axis=1)

            # ROC（Rate of Change）是一个常用的技术指标，用于衡量当前价格与之前某一价格之间的变化率。
            # 它的计算公式为：ROC = ((price/prevPrice) - 1) * 100。这个指标可以帮助投资者判断价格的变化速度和趋势。
            indicator = talib.ROC(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_ROC_{period}')], axis=1)

            indicator = talib.ROCP(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_ROCP_{period}')], axis=1)

            indicator = talib.ROCR(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_ROCR_{period}')], axis=1)

            # - RSI 的值在 0 到 100 之间波动。
            #    - 通常，RSI 高于 70 被视为超买状态，可能意味着价格即将回调。
            #    - RSI 低于 30 被视为超卖状态，可能意味着价格即将反弹。
            indicator = talib.RSI(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_RSI_{period}')], axis=1)

            # 标准差是一种统计学指标，用于衡量数据的离散程度。在金融领域，标准差常用于衡量股票价格的波动性。较高的标准差表示价格波动较大，而较低的标准差则表示价格相对稳定。
            indicator = talib.STDDEV(df['close'], timeperiod=self.periods[period])
            indicator.fillna(0, inplace=True)
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_STDDEV_{period}')], axis=1)

            # T3 指标是一种平滑的移动平均线，它通过多次应用指数移动平均线（EMA）来减少价格波动的影响，从而提供更平滑的趋势线。
            indicator = talib.T3(df['close'], timeperiod=self.periods[period])
            indicator.fillna(0, inplace=True)
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_T3_{period}')], axis=1)

            # 当TRIX从负值转为正值时，可能是买入信号；反之，当TRIX从正值转为负值时，可能是卖出信号。
            indicator = talib.TRIX(df['close'], timeperiod=self.periods[period])
            indicator.fillna(0, inplace=True)
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_TRIX_{period}')], axis=1)

            # 时间序列预测指标，它通过线性回归方法预测未来的价格走势。TSF 指标的计算基于过去一段时间的价格数据，通常用于识别价格趋势并预测未来的价格变化。
            indicator = talib.TSF(df['close'], timeperiod=self.periods[period])
            indicator.fillna(0, inplace=True)
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_TSF_{period}')], axis=1)

            # VAR（方差）是一个常用的统计指标，用于衡量数据集的波动性。方差越大，数据的波动性越大；方差越小，数据的波动性越小。在金融市场中，方差可以帮助投资者了解价格的波动情况，从而更好地进行风险管理和投资决策。
            indicator = talib.VAR(df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_VAR_{period}')], axis=1)

            # Williams %R 是一种动量指标，用于衡量市场的超买或超卖状态。
            # 它通过比较特定时间段内的收盘价与价格高低区间来计算。其值在 -100 到 0 之间波动，通常用于识别潜在的反转点。
            # 接近 -100：表示超卖（价格接近最低价）。
            # 接近 0：表示超买（价格接近最高价）。
            indicator = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=self.periods[period])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_WILLR_{period}')], axis=1)

        # 加权收盘价（Weighted Close Price, WCLPRICE），我们可以使用 `talib` 库中的 `WCLPRICE` 函数。
        # 加权收盘价是一个技术分析指标，它通过对开盘价、最高价、最低价和收盘价进行加权平均来反映股票的真实价格水平。
        indicator = talib.WCLPRICE(df['high'], df['low'], df['close'])
        df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_WCLPRICE')], axis=1)

        # TYPPRICE（Typical Price，典型价格）是技术分析中常用的一个指标，用于计算某一时间段内价格的平均水平。它通过将最高价、高价和最低价相加后除以3来得出。
        # TYPPRICE可以帮助交易者更好地理解市场的平均价格水平，从而辅助判断市场趋势和价格波动。
        indicator = talib.TYPPRICE(df['high'], df['low'], df['close'])
        df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_TYPPRICE')], axis=1)

        # True Range 是技术分析中用来衡量市场波动性的指标。它考虑了当前最高价、最低价和前一个收盘价之间的最大差值。
        indicator = talib.TRANGE(df['high'], df['low'], df['close'])
        df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_TRANGE')], axis=1)

        '''
        1. **超买超卖判断**：通常情况下，%K 和 %D 低于 20 被认为是超卖状态，高于 80 被认为是超买状态。
        2. **交叉信号**：当 %K 线从下向上穿过 %D 线时，可能是买入信号；反之，当 %K 线从上向下穿过 %D 线时，可能是卖出信号。
        3. **背离分析**：如果价格创出新高或新低，而 STOCHF 指标没有相应创出新高或新低，可能预示着趋势反转。
        '''
        stochf_k, stochf_d = talib.STOCHF(df['high'], df['low'], df['close'])
        gold_cross, death_cross = IndicatorService.calculate_stochf_crosses(stochf_k, stochf_d)
        df = pd.concat([df, pd.Series(stochf_k, name=f'indicator_talib_stochf_k')], axis=1)
        df = pd.concat([df, pd.Series(stochf_d, name=f'indicator_talib_stochf_d')], axis=1)
        df = pd.concat([df, pd.Series(gold_cross, name=f'indicator_talib_stochf_gold_cross')], axis=1)
        df = pd.concat([df, pd.Series(death_cross, name=f'indicator_talib_stochf_death_cross')], axis=1)

        # - SAR 是一种趋势跟踪指标，用于识别价格的反转点。
        # SAR指标可以帮助识别当前市场的趋势方向。当价格在SAR点之上时，通常被视为上升趋势；当价格在SAR点之下时，通常被视为下降趋势。
        # SAR指标也可以用作动态止损点。当价格反转并穿过SAR点时，可能意味着趋势的反转，交易者可以考虑平仓或反向操作。
        indicator = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_SAR')], axis=1)

        # 交易者可以使用 SAREXT 来设置止损点或识别趋势反转。通常，当价格从下方穿过 SAREXT 点时，表明可能的买入信号；
        # 当价格从上方穿过 SAREXT 点时，表明可能的卖出信号。
        indicator = talib.SAREXT(df['high'], df['low'])
        df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_SAREXT')], axis=1)

        # NATR 是 ATR 的标准化版本，ATR 是一种衡量市场波动性的指标。通过将 ATR 标准化为百分比形式，
        # NATR 提供了一个更直观的波动性衡量标准，尤其是在不同价格水平的市场中。
        indicator = talib.NATR(df['high'], df['low'], df['close'])
        df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_NATR')], axis=1)

        # OBV（On Balance Volume，平衡交易量）是一个技术分析指标，用于衡量交易量的变化与价格变化之间的关系。
        # 它通过累积每日的交易量来反映市场的买卖压力。具体来说，当收盘价高于前一天的收盘价时，OBV增加当天的交易量；
        # 当收盘价低于前一天的收盘价时，OBV减少当天的交易量；如果收盘价与前一天持平，则OBV不变。
        # 当 OBV 和价格走势出现背离时，可能预示着趋势的反转。
        # 例如，如果价格在上涨而 OBV 在下降，这可能表明买入压力正在减弱，价格可能会下跌。反之亦然。
        indicator = talib.OBV(df['close'], df['indicator_vol'])
        df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_OBV')], axis=1)

        df['indicator_talib_avgprice'] = talib.AVGPRICE(df['open'], df['high'], df['low'], df['close'])
        df['indicator_talib_BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])

        # 希尔伯特变换-主导周期周期（Hilbert Transform - Dominant Cycle Period）。
        # 该函数基于希尔伯特变换技术，用于识别价格序列中的主导周期（Dominant Cycle），即当前市场中最显著的价格波动周期。
        indicator = talib.HT_DCPERIOD(df['close'])
        df = pd.concat([df, pd.Series(indicator, name='indicator_talib_HT_DCPERIOD')], axis=1)
        indicator = talib.HT_DCPHASE(df['close'])
        df = pd.concat([df, pd.Series(indicator, name='indicator_talib_HT_DCPHASE')], axis=1)

        # HT_PHASOR 是一种技术指标，用于分析市场周期和趋势。它通过希尔伯特变换来计算相位分量，通常用于识别市场的周期性变化。
        inphase, quadrature = talib.HT_PHASOR(df['close'])
        df = pd.concat([df, pd.Series(inphase, name='indicator_talib_inphase')], axis=1)
        df = pd.concat([df, pd.Series(quadrature, name='indicator_talib_quadrature')], axis=1)

        # 主要用于识别市场周期和趋势反转点。它通过计算正弦波和余弦波来帮助识别价格的周期性变化。
        # 通过计算正弦波（sine）和余弦波（leadsine），帮助识别市场的周期性变化。通常，当正弦波和余弦波交叉时，可能预示着趋势的反转。
        sine, leadsine = talib.HT_SINE(df['close'])
        df = pd.concat([df, pd.Series(sine, name='indicator_talib_sine')], axis=1)
        df = pd.concat([df, pd.Series(leadsine, name='indicator_talib_leadsine')], axis=1)

        # Instantaneous Trendline（HT_TRENDLINE）指标，我们可以使用 `talib` 库。这个指标是通过希尔伯特变换来计算的，用于识别价格趋势的变化
        indicator = talib.HT_TRENDLINE(df['close'])
        indicator.fillna(0, inplace=True)
        df = pd.concat([df, pd.Series(indicator, name='indicator_talib_HT_TRENDLINE')], axis=1)

        # 识别市场是处于趋势模式还是循环模式。
        # - `HT_TRENDMODE` 指标返回一个二进制值，通常为 1 或 0。
        #    - 当返回值为 1 时，表示市场处于趋势模式。
        #    - 当返回值为 0 时，表示市场处于循环模式。
        indicator = talib.HT_TRENDMODE(df['close'])
        df = pd.concat([df, pd.Series(indicator, name='indicator_talib_HT_TRENDMODE')], axis=1)

        mama, fama = talib.MAMA(df['close'])
        df = pd.concat([df, pd.Series(mama, name=f'indicator_talib_MAMA')], axis=1)
        df = pd.concat([df, pd.Series(fama, name=f'indicator_talib_FAMA')], axis=1)

        # CDL2CROWS（Two Crows）是一个由三根K线组成的看跌反转形态。它通常出现在上升趋势中，预示着可能的趋势反转。
        # 具体来说，它由一根长阳线和随后的两根短阴线组成，其中第二根阴线的开盘价在第一根阳线的实体之上，且第二根阴线的收盘价低于第一根阳线的收盘价。
        indicator = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
        df = pd.concat([df, pd.Series(indicator, name='indicator_talib_CDL2CROWS')], axis=1)

        # `CDL3BLACKCROWS` 是一个用于识别三只乌鸦形态的技术指标。
        # 三只乌鸦形态是一种看跌反转形态，通常出现在上升趋势的末端，由连续三根长阴线组成，每根阴线的收盘价都低于前一根阴线的收盘价。
        df['indicator_talib_CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])

        '''
        `CDL3INSIDE` 是一个用于识别三内部上涨/下跌形态的技术指标。这个形态通常出现在市场趋势的反转点，包含三根连续的K线。具体来说：
        1. 第一根K线是长实体，表示当前趋势的延续。
        2. 第二根K线是小实体，完全包含在第一根K线的范围内，表示市场犹豫。
        3. 第三根K线的收盘价超越了第一根K线的开盘价，确认了趋势的反转。
        '''
        df['indicator_talib_CDL3INSIDE'] = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])

        # `CDL3LINESTRIKE` 是一个蜡烛图形态识别指标，用于识别三线打击形态（Three Line Strike）。在技术分析中，三线打击形态是一种反转形态，通常出现在上升趋势或下降趋势的末端。
        df['indicator_talib_CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])

        # `CDL3STARSINSOUTH` 是一个用于识别三颗星形态的技术指标。三颗星形态是一种罕见的看涨反转形态，通常出现在下跌趋势的末期。
        # 它由三根连续的小实体蜡烛组成，其中每根蜡烛的最低价逐渐升高。
        df['indicator_talib_CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])

        # `CDL3WHITESOLDIERS` 是一个用于识别三只白兵形态的技术指标。
        # 三只白兵形态是一种看涨的反转形态，通常出现在下跌趋势的末端，由连续三根长阳线组成，每根阳线的收盘价都高于前一根阳线的收盘价，并且开盘价在前一根阳线的实体之内。
        df['indicator_talib_CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])

        '''
        CDL_ABANDONED_BABY 是一个由三根蜡烛组成的反转形态指标，通常用于识别市场趋势的反转。它由以下几个部分组成：
        1. **第一根蜡烛**：在下跌趋势中是长阴线，在上涨趋势中是长阳线。
        2. **第二根蜡烛**：是一根十字星（开盘价和收盘价几乎相同），并且与前一根蜡烛之间有一个价格缺口。
        3. **第三根蜡烛**：在下跌趋势中是长阳线，在上涨趋势中是长阴线，并且与第二根蜡烛之间也有一个价格缺口。
        这个形态表明市场可能会发生反转。在下跌趋势中出现时，可能预示着市场将转为上涨；在上涨趋势中出现时，可能预示着市场将转为下跌。
        '''
        df['indicator_talib_abandoned_baby'] = talib.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close'])

        # `CDLADVANCEBLOCK`（高级阻力）指标，我们可以使用`talib`库中的`CDLADVANCEBLOCK`函数。
        # 这个函数用于识别K线图中的高级阻力形态，它是一种反转形态，通常出现在上升趋势的末期。
        df['indicator_talib_CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close'])

        # Belt-hold (捉腰带线) 指标，我们可以使用 `talib` 库中的 `CDLBELTHOLD` 函数。
        # 这个函数会根据开盘价、最高价、最低价和收盘价来识别捉腰带线形态，并返回一个包含形态识别结果的数组。
        df['indicator_talib_CDLBELTHOLD'] = talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])

        # `CDLBREAKAWAY` 是一个用于识别市场反转形态的函数，它属于蜡烛图形态分析的一部分。Breakaway 形态是一种由五根蜡烛组成的形态，通常出现在市场趋势的末端，可能预示着趋势的反转。这个形态可以是看涨的，也可以是看跌的。
        df['indicator_talib_breakaway'] = talib.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close'])

        #
        df['indicator_talib_closingmarubozu'] = talib.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close'])

        # CDLCONCEALBABYSWALL 是一个用于识别 K 线形态的技术指标，属于蜡烛图形态分析的一部分。
        # 它用于识别市场中的潜在反转信号，特别是在下跌趋势中可能出现的看涨反转信号。这个形态由四根特定的蜡烛线组成，通常出现在市场的底部。
        df['indicator_talib_cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close'])

        # CDLCOUNTERATTACK 是一种由两根K线组成的反转形态，通常出现在市场的顶部或底部。
        # 它表示市场在一个方向上强烈移动后，第二天价格回到前一天的收盘价附近，显示出市场可能会反转的信号。
        df['indicator_talib_CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close'])

        # CDLDARKCLOUDCOVER 是一个由日本蜡烛图技术衍生出来的技术指标，中文称为“乌云盖顶”。
        # 它是一种看跌反转形态，通常出现在上升趋势的顶部，预示着市场可能会从上升趋势转为下降趋势。
        df['indicator_talib_CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])

        # Doji 是一种常见的蜡烛图形态，通常表示市场的不确定性或潜在的反转信号。它的特征是开盘价和收盘价非常接近，形成一个十字形态。
        df['indicator_talib_CDLDOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])

        # CDLDOJISTAR 是一个用于识别蜡烛图形态的技术指标，属于形态识别类指标。
        # Doji Star（十字星）是一种常见的反转形态，通常出现在市场趋势的末端，可能预示着趋势的反转。
        # 它由两根蜡烛组成，第一根蜡烛是一个长实体，第二根蜡烛是一个十字星，表示市场的不确定性。
        df['indicator_talib_CDLDOJISTAR'] = talib.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close'])

        # Dragonfly Doji 是一种蜡烛图形态，通常出现在市场底部，表示潜在的趋势反转信号。
        # 它的特征是开盘价、收盘价和最高价几乎相同，而最低价明显低于这三者，形成一个“蜻蜓”形状。
        df['indicator_talib_CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])

        # CDLENGULFING（吞噬形态）是技术分析中一种常用的K线形态，用于识别市场的反转信号。
        # 它由两根K线组成，第二根K线的实体完全包住第一根K线的实体。根据形态的不同，吞噬形态可以分为看涨吞噬和看跌吞噬。
        df['indicator_talib_CDLENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])

        # 这个指标用于识别市场的潜在反转信号，特别是在上升趋势的末期。
        df['indicator_talib_evening_doji_star'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])

        # Evening Star（黄昏星）形态，我们可以使用 `talib` 库中的 `CDLEVENINGSTAR` 函数。
        # 黄昏星是一种看跌的反转形态，通常出现在上升趋势的顶部。
        # 它由三根蜡烛组成：第一根是长阳线，第二根是小实体（可以是阳线或阴线），第三根是长阴线。
        df['indicator_talib_eveningstar'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])

        # CDL_GAPSIDESIDEWHITE 是一个由日本蜡烛图技术分析中识别的形态，称为“上/下跳空并列白色线形态”。
        # 这种形态通常出现在市场趋势的延续阶段，可能预示着当前趋势的持续。
        # 它通过分析开盘价、最高价、最低价和收盘价来识别这种形态，并返回一个整数值来表示信号的强度和方向。
        # 通常，返回值为正数表示看涨信号，负数表示看跌信号，零表示没有信号。
        df['indicator_talib_CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close'])

        # Gravestone Doji 蜡烛图形态指标，我们可以使用 `talib` 库中的 `CDLGRAVESTONEDOJI` 函数。
        # Gravestone Doji 是一种技术分析中的蜡烛图形态，通常被视为市场反转的信号。
        # 它的特征是开盘价、收盘价和最低价接近相同，而最高价明显高于这三个价格。
        df['indicator_talib_gravestone_doji'] = talib.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])

        # 函数用于识别锤子线形态，锤子线是一种常见的反转形态，通常出现在下跌趋势的底部，具有较长的下影线和较短的实体。
        df['indicator_talib_CDLHAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])

        df['indicator_talib_CDLHANGINGMAN'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])

        # Harami Pattern（母子线形态），我们可以使用 `talib` 库中的 `CDLHARAMI` 函数。母子线形态是一种常见的反转形态，通常出现在市场趋势的末端，表示市场可能会发生反转。
        df['indicator_talib_CDLHARAMI'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])

        # CDL_HARAMICROSS 是一个由两根蜡烛组成的反转形态指标，属于蜡烛图技术分析的一部分。它通常出现在市场趋势的末端，可能预示着趋势的反转。
        # 具体来说，第一根蜡烛是大实体，第二根蜡烛是十字星，且十字星的实体完全包含在前一根蜡烛的实体之内。
        df['indicator_talib_CDL_HARAMICROSS'] = talib.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])

        # High-Wave Candle 是一种蜡烛图形态，通常表示市场的不确定性或潜在的反转信号。它的特征是具有长上下影线和较短的实体。
        df['indicator_talib_CDLHIGHWAVE'] = talib.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close'])

        # Hikkake Pattern）是一个由日本蜡烛图技术分析中识别的形态。
        # 它通常用于识别市场的反转信号。
        # Hikkake Pattern 是一种短期的价格形态，通常由六根蜡烛组成，前三根形成一个内包线，接下来的三根则是突破和确认。
        df['indicator_talib_CDLHIKKAKE'] = talib.CDLHIKKAKE(df['open'], df['high'], df['low'], df['close'])
        df['indicator_talib_CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close'])

        # Homing Pigeon 是一个蜡烛图形态指标，用于识别潜在的市场反转信号。
        # 它通常出现在下跌趋势中，由两根蜡烛组成。
        # 第一根蜡烛是长阴线，第二根蜡烛是短阳线，且第二根蜡烛的实体完全包含在第一根蜡烛的实体之内。这个形态表明市场可能会出现反转，价格可能会上涨。
        df['indicator_talib_CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close'])

        # Identical Three Crows）是一个由三根连续的阴线组成的看跌反转形态。
        # 它通常出现在上升趋势的顶部，预示着趋势可能反转向下。
        # 每根阴线的开盘价都接近前一根阴线的收盘价，并且每根阴线的收盘价都低于前一根阴线的收盘价。
        df['indicator_talib_CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['close'])

        # （In-Neck Pattern）是由日本蜡烛图技术分析中的一种形态识别指标。它用于识别市场中的反转信号，特别是在下跌趋势中的潜在反转。
        df['indicator_talib_CDLINNECK'] = talib.CDLINNECK(df['open'], df['high'], df['low'], df['close'])

        #  **CDLINVERTEDHAMMER（倒锤子线）**：这是一个单根K线形态，通常出现在下跌趋势中，可能预示着趋势反转。
        #  倒锤子线的特征是有一个较长的上影线和较短的实体，实体在当天的价格区间的下端。
        df['indicator_talib_CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])

        # （Kicking）是由两根蜡烛线组成的反转形态，通常出现在市场的顶部或底部。
        # 它由一根长长的白色蜡烛和一根长长的黑色蜡烛组成，且两根蜡烛之间没有影线（即跳空）。
        # 这个形态表明市场情绪的突然转变，可能预示着趋势的反转。
        df['indicator_talib_CDLKICKING'] = talib.CDLKICKING(df['open'], df['high'], df['low'], df['close'])

        '''
        CDL_KICKINGBYLENGTH 是一个由两根蜡烛线组成的形态指标，用于识别市场的反转信号。
        它属于蜡烛图形态分析的一部分，通常用于判断市场的多头（bullish）或空头（bearish）趋势。
        - **Kicking Bullish**: 当第二根蜡烛线是长阳线（开盘价低于收盘价），并且比第一根蜡烛线（长阴线，开盘价高于收盘价）更长时，表明市场可能从空头转为多头。
        - **Kicking Bearish**: 当第二根蜡烛线是长阴线，并且比第一根蜡烛线（长阳线）更长时，表明市场可能从多头转为空头。
        '''
        df['indicator_talib_CDL_KICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close'])

        '''
        （Ladder Bottom）是一个由五根K线组成的看涨反转形态，通常出现在下跌趋势的末端。
        这个形态的特征是前三根K线为阴线，第四根K线为小阴线或小阳线，第五根K线为大阳线，且收盘价高于前四根K线的开盘价。
        '''
        df['indicator_talib_CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close'])

        # （长腿十字星）**：这是一个蜡烛图形态指标，用于识别市场中的潜在反转信号。
        # 长腿十字星是一种特殊的十字星形态，具有长长的上下影线，表示市场在开盘和收盘之间经历了剧烈的波动，但最终收盘价与开盘价几乎相同。
        # 这种形态通常出现在市场趋势的末端，可能预示着趋势的反转。
        df['indicator_talib_CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])

        '''
        `CDLLONGLINE` 是一个用于识别长蜡烛线的技术指标。长蜡烛线通常表示市场中存在较强的买卖力量，可能预示着趋势的延续或反转。
        - `CDLLONGLINE` 返回一个整数值，通常为 -100, 0 或 100。
       - `100` 表示出现了看涨的长蜡烛线。
       - `-100` 表示出现了看跌的长蜡烛线。
       - `0` 表示没有检测到长蜡烛线。
        '''
        df['indicator_talib_CDLLONGLINE'] = talib.CDLLONGLINE(df['open'], df['high'], df['low'], df['close'])

        # Marubozu 是一种没有上下影线的蜡烛图形态，表示市场在开盘后一直朝一个方向运动，直到收盘。根据蜡烛的颜色，Marubozu 可以分为两种：
        # 1. **白色 Marubozu**：开盘价等于最低价，收盘价等于最高价，表示强烈的看涨信号。
        # 2. **黑色 Marubozu**：开盘价等于最高价，收盘价等于最低价，表示强烈的看跌信号。
        df['indicator_talib_marubozu'] = talib.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])

        #
        df['indicator_talib_CDL_MATCHINGLOW'] = talib.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['close'])

        # 用于识别“持仓模式”（Mat Hold）形态。这是一种反转形态，通常出现在上升趋势中，表示市场可能会继续上涨。
        # 该形态由五根蜡烛线组成，通常在第三根蜡烛线出现回调后，第四和第五根蜡烛线继续上涨。
        df['indicator_talib_cdlmathold'] = talib.CDLMATHOLD(df['open'], df['high'], df['low'], df['close'])

        # `CDLMORNINGDOJISTAR` 是一个蜡烛图形态识别指标，用于识别市场中的晨星十字星形态，这是一种潜在的看涨反转信号。该指标返回一个整数值，表示识别到的形态信号。
        df['indicator_talib_CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])

        # `CDLMORNINGSTAR`（晨星形态）指标，我们可以使用 `talib` 库。
        # `CDLMORNINGSTAR` 是一种常用的蜡烛图形态分析工具，用于识别潜在的市场反转信号。
        # 它通常出现在下跌趋势的末端，预示着可能的上涨反转。
        df['indicator_talib_CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])

        # `CDLONNECK` 是一个用于识别“颈上线”形态的技术指标函数，属于蜡烛图形态分析的一部分。这个形态通常出现在下跌趋势中，可能预示着趋势反转或短期的价格回升。
        # 它由两根蜡烛组成，第一根是长阴线，第二根是开盘价接近前一根蜡烛的收盘价，并且收盘价接近最低价的小阳线或小阴线。
        df['indicator_talib_CDLONNECK'] = talib.CDLONNECK(df['open'], df['high'], df['low'], df['close'])

        # Piercing Pattern（刺透形态）指标，我们可以使用 `talib` 库中的 `CDLPIERCING` 函数。这个函数用于识别蜡烛图中的刺透形态，这是一种潜在的看涨反转信号。
        df['indicator_talib_CDLPIERCING'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])

        '''
        “人力车夫”形态。这是一种蜡烛图形态，通常用于判断市场的潜在反转信号。
        - 该指标返回的值通常为 0 或者非零值。0 表示没有识别出形态，非零值表示识别出“人力车夫”形态。
        - 正值通常表示看涨的反转信号，而负值表示看跌的反转信号。
        '''
        df['indicator_talib_CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['close'])

        # `CDLRISEFALL3METHODS` 是一个用于识别“上升/下降三法”形态的技术指标。
        # 这个形态属于蜡烛图形态分析的一部分，通常用于判断市场的持续趋势。
        # 上升三法是一个看涨的持续形态，而下降三法是一个看跌的持续形态。
        df['indicator_talib_CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close'])

        # `CDLSEPARATINGLINES` 是一个用于识别分离线形态的技术指标。分离线形态是一种反转形态，通常出现在市场趋势的末端。
        # 它由两根K线组成，第一根K线与当前趋势方向一致，而第二根K线则与第一根K线的开盘价相同，但方向相反。
        # 通常为 `100`、`-100` 或 `0`。`100` 表示看涨的分离线形态，`-100` 表示看跌的分离线形态，而 `0` 则表示没有识别出分离线形态。
        df['indicator_talib_CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['close'])

        # `CDLSHOOTINGSTAR` 是 TA-Lib 库中的一个函数，用于识别“射击之星”形态。
        # 这是一种常见的反转形态，通常出现在上升趋势的顶部，预示着可能的趋势反转。
        # 射击之星形态的特征是有一个较长的上影线、小实体和很小或没有下影线。
        df['indicator_talib_shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])

        # CDLSHORTLINE - 短蜡烛 提示市场可能进入盘整或反转
        df['indicator_talib_CDLSHORTLINE'] = talib.CDLSHORTLINE(df['open'], df['high'], df['low'], df['close'])

        # 纺锤线是一种常见的K线形态，通常表示市场的不确定性，可能预示着趋势的反转或持续。它的特征是实体较小，上下影线较长。
        # 正值表示看涨的纺锤线，负值表示看跌的纺锤线，零值表示没有纺锤线形态。
        df['indicator_talib_CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])

        # 停滞形态（Stalled Pattern），这是一种反转形态，通常出现在上升趋势中，预示着市场可能会出现反转。
        df['indicator_talib_CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['close'])

        '''
        `CDLSTICKSANDWICH` 是一个用于识别股票或其他金融市场中“夹心线”形态的技术指标。这个形态通常出现在下跌趋势中，可能预示着市场的反转。
        1. 第一根蜡烛线是长阴线，表示市场的强烈下跌。
        2. 第二根蜡烛线是阳线，开盘价低于前一根阴线的收盘价，但收盘价接近或等于第一根阴线的收盘价。
        3. 第三根蜡烛线是阳线，开盘价高于第二根阳线的开盘价，收盘价高于第二根阳线的收盘价。
        '''
        df['indicator_talib_CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['close'])

        # `CDLTAKURI` 是其中一个用于识别蜡烛图形态的函数，专门用于检测 Takuri（长下影线的蜻蜓十字星）形态。
        # 这种形态通常出现在下跌趋势的末端，可能预示着市场的反转。
        df['indicator_talib_CDLTAKURI'] = talib.CDLTAKURI(df['open'], df['high'], df['low'], df['close'])

        #
        '''
        Tasuki Gap 是一种趋势延续形态，通常出现在上升或下降趋势中，表明市场可能会继续沿着当前趋势方向发展。
        上升趋势中的 Tasuki Gap：
            第一根K线是一根大阳线，表明上涨趋势强劲。
            第二根K线向上跳空（开盘价高于前一根K线的最高价），并继续收阳。
            第三根K线是一根阴线，收盘价位于第二根K线的实体内，但未完全填补跳空缺口。
        下降趋势中的 Tasuki Gap：
            第一根K线是一根大阴线，表明下跌趋势强劲。
            第二根K线向下跳空（开盘价低于前一根K线的最低价），并继续收阴。
            第三根K线是一根阳线，收盘价位于第二根K线的实体内，但未完全填补跳空缺口。
        100：表示识别到上升趋势中的 Tasuki Gap。
        -100：表示识别到下降趋势中的 Tasuki Gap。
        0：表示未识别到 Tasuki Gap 形态。
        '''
        df['indicator_talib_CDLTASUKIGAP'] = talib.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['close'])

        '''
        “插入线”（Thrusting Pattern）K线形态。插入线是一种潜在的反转信号，通常出现在下跌趋势中，表明市场可能即将反转向上。
        下跌趋势中的插入线：
            第一根K线是一根大阴线，表明下跌趋势强劲。
            第二根K线是一根阳线，开盘价低于前一根K线的最低价，但收盘价位于前一根K线实体的中部以下。
            第二根K线的收盘价未能突破前一根K线的中点，表明多头力量不足。
        形态的关键点：
            第二根K线的收盘价位于第一根K线实体的下半部分。
            第二根K线的实体较短，表明多头力量较弱。
        '''
        df['indicator_talib_CDLTHRUSTING'] = talib.CDLTHRUSTING(df['open'], df['high'], df['low'], df['close'])

        # 三星形态是一种罕见但重要的反转信号，通常出现在市场顶部或底部，表明趋势可能即将反转。
        df['indicator_talib_CDLTRISTAR'] = talib.CDLTRISTAR(df['open'], df['high'], df['low'], df['close'])

        # “独特三川形态”（Unique 3 River）K线形态。这是一种较为复杂的反转形态，通常出现在下跌趋势中，可能预示着趋势的反转或市场的底部形成。
        # 通常，返回值为 0 表示没有识别出形态，正值表示看涨反转形态，负值表示看跌反转形态。
        df['indicator_talib_CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['close'])

        #
        # 向上跳空两只乌鸦”。这种形态通常出现在上升趋势中，可能预示着趋势反转的信号。
        # -100：表示识别到 Upside Gap Two Crows 形态。
        # 0：表示未识别到 Upside Gap Two Crows 形态。
        df['indicator_talib_CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['close'])

        '''
        “上升/下降跳空三法”（Upside/Downside Gap Three Methods）K线形态。这是一种趋势延续形态，通常出现在上升或下降趋势中，表明市场可能会继续沿着当前趋势方向发展。
        上升跳空三法（Upside Gap Three Methods）：
        出现在上升趋势中。
            第一根K线：一根大阳线，表明上升趋势强劲。
            第二根K线：一根小阴线，开盘价高于第一根K线的收盘价（向上跳空），但收盘价低于第一根K线的收盘价。
            第三根K线：一根大阳线，收盘价高于第一根K线的收盘价，表明上升趋势继续。
        下降跳空三法（Downside Gap Three Methods）：
        出现在下降趋势中。
            第一根K线：一根大阴线，表明下降趋势强劲。
            第二根K线：一根小阳线，开盘价低于第一根K线的收盘价（向下跳空），但收盘价高于第一根K线的收盘价。
            第三根K线：一根大阴线，收盘价低于第一根K线的收盘价，表明下降趋势继续。
        '''
        df['indicator_talib_CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['close'])

        # Bollinger Bands (布林带) 指标，我们可以使用 `talib` 库中的 `BBANDS` 函数。
        for period in self.bbands_period:
            params = self.bbands_period[period]
            upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=params[0], nbdevup=params[1], nbdevdn=params[2], matype=0)
            # 将结果添加到 DataFrame 中
            df[f'indicator_talib_upperband_{period}'] = upperband
            df[f'indicator_talib_middleband_{period}'] = middleband
            df[f'indicator_talib_lowerband_{period}'] = lowerband

        for period in self.macd_period:
            params = self.macd_period[period]
            macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=params[0], slowperiod=params[1], signalperiod=params[2])
            macd.fillna(0, inplace=True)
            macdsignal.fillna(0, inplace=True)
            macdhist.fillna(0, inplace=True)

            gold_cross, death_cross = IndicatorService.calculate_macd_cross(macd, macdsignal)

            df = pd.concat([df, pd.Series(macd, name=f'indicator_talib_macd_{period}')], axis=1)
            df = pd.concat([df, pd.Series(macdsignal, name=f'indicator_talib_macdsignal_{period}')], axis=1)
            df = pd.concat([df, pd.Series(macdhist, name=f'indicator_talib_macdhist_{period}')], axis=1)
            df = pd.concat([df, pd.Series(gold_cross, name=f'indicator_talib_macd_gold_cross_{period}')], axis=1)
            df = pd.concat([df, pd.Series(death_cross, name=f'indicator_talib_macd_death_cross_{period}')], axis=1)

            # PPO可以帮助交易者识别价格趋势的强度和方向。当PPO为正值时，表示短期平均线在长期平均线之上，表明市场处于上升趋势。反之，当PPO为负值时，表示短期平均线在长期平均线之下，表明市场处于下降趋势。
            # PPO常用于识别趋势反转点、确认趋势方向以及识别超买或超卖状态。由于PPO是以百分比表示的，因此在不同价格水平的资产之间进行比较时非常有用。
            ppo = talib.PPO(df['close'], fastperiod=params[0], slowperiod=params[1], matype=0)
            df = pd.concat([df, pd.Series(ppo, name=f'indicator_talib_PPO_{period}')], axis=1)

            apo = talib.APO(df['close'], fastperiod=params[0], slowperiod=params[1], matype=0)
            df = pd.concat([df, pd.Series(apo, name=f'indicator_talib_APO_{period}')], axis=1)

            '''
            0   SMA (Simple Moving Average) 简单移动平均线，对所有数据点赋予相同权重。
            1   EMA (Exponential Moving Average)    指数移动平均线，对近期数据点赋予更高权重。
            2   WMA (Weighted Moving Average)   加权移动平均线，对近期数据点赋予线性递增的权重。
            3   DEMA (Double Exponential Moving Average)    双指数移动平均线，对近期数据点赋予更高的权重，反应更快。
            4   TEMA (Triple Exponential Moving Average)    三指数移动平均线，对近期数据点赋予更高的权重，反应更快。
            5   TRIMA (Triangular Moving Average)   三角移动平均线，对中间数据点赋予更高权重。
            6   KAMA (Kaufman Adaptive Moving Average)  自适应移动平均线，根据市场波动性动态调整权重。
            7   MAMA (MESA Adaptive Moving Average) MESA 自适应移动平均线，基于频率分析动态调整权重。
            8   T3 (Triple Exponential Moving Average)  三重指数移动平均线，对近期数据点赋予更高的权重，反应更快。
            '''
            for ma_type in range(0, 9):
                macd, macdsignal, macdhist = talib.MACDEXT(
                    df['close'],
                    fastperiod=params[0],
                    fastmatype=ma_type,
                    slowperiod=params[1],
                    slowmatype=ma_type,
                    signalperiod=params[2],
                    signalmatype=ma_type
                )
                macd.fillna(0, inplace=True)
                macdsignal.fillna(0, inplace=True)
                macdhist.fillna(0, inplace=True)
                gold_cross, death_cross = IndicatorService.calculate_macd_cross(macd, macdsignal)

                df = pd.concat([df, pd.Series(macd, name=f'indicator_talib_macd_ext_type{ma_type}_{period}')], axis=1)
                df = pd.concat([df, pd.Series(macdsignal, name=f'indicator_talib_macdsignal_ext_type{ma_type}_{period}')], axis=1)
                df = pd.concat([df, pd.Series(macdhist, name=f'indicator_talib_macdhist_ext_type{ma_type}_{period}')], axis=1)
                df = pd.concat([df, pd.Series(gold_cross, name=f'indicator_talib_macd_gold_cross_ext_type{ma_type}_{period}')], axis=1)
                df = pd.concat([df, pd.Series(death_cross, name=f'indicator_talib_macd_death_cross_ext_type{ma_type}_{period}')], axis=1)

        for period in self.kdj_period:
            params = self.kdj_period[period]

            for ma_type in range(0, 9):
                stoch_k, stoch_d = talib.STOCH(
                    df['high'],
                    df['low'],
                    df['close'],
                    fastk_period=params[0],  # 快速K线的时间周期
                    slowk_period=params[1],  # 慢速K线的时间周期
                    slowk_matype=ma_type,  # 慢速K线的移动平均类型
                    slowd_period=params[2],  # 慢速D线的时间周期
                    slowd_matype=ma_type  # 慢速D线的移动平均类型
                )
                stoch_k.fillna(0, inplace=True)
                stoch_d.fillna(0, inplace=True)
                df = pd.concat([df, pd.Series(stoch_k, name=f'indicator_talib_stoch_k_type_{ma_type}_{period}')], axis=1)
                df = pd.concat([df, pd.Series(stoch_d, name=f'indicator_talib_stoch_d_type_{ma_type}_{period}')], axis=1)

                # STOCHRSI 的值通常在 0 到 1 之间波动。当 STOCHRSI 高于 0.8 时，市场可能处于超买状态；当低于 0.2 时，市场可能处于超卖状态。这些信息可以帮助交易者做出买卖决策。
                fastk, fastd = talib.STOCHRSI(df['close'], timeperiod=params[0], fastk_period=params[1], fastd_period=params[2], fastd_matype=ma_type)
                fastk.fillna(0, inplace=True)
                fastd.fillna(0, inplace=True)
                df = pd.concat([df, pd.Series(fastk, name=f'indicator_talib_stochrsi_fastk_{ma_type}_{period}')], axis=1)
                df = pd.concat([df, pd.Series(fastd, name=f'indicator_talib_stochrsi_fastd_{ma_type}_{period}')], axis=1)

        for period in self.ultosc_period:
            params = self.ultosc_period[period]

            # Ultimate Oscillator（ULTOSC）是一个技术分析指标，用于衡量市场的买卖压力。它结合了不同时间周期的动量，旨在减少传统振荡指标的缺陷，如过于敏感或滞后。
            # ULTOSC通过结合短期、中期和长期的动量来提供更平滑和可靠的信号。
            indicator = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=params[0], timeperiod2=params[1], timeperiod3=params[2])
            df = pd.concat([df, pd.Series(indicator, name=f'indicator_talib_ULTOSC_{period}')], axis=1)

        return df
