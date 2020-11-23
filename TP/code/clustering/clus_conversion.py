import numpy as np
import pandas as pd
import seaborn as sns

country = pd.read_csv("data/Country.csv")
indicators = pd.read_csv("data/Indicators.csv")

chosen_indicators = [
    'EN.ATM.CO2E.KT',  # CO2 emissions (kt)
    'EN.ATM.CO2E.GF.ZS',  # CO2 emissions from gaseous fuel consumption (% of total)
    # 'EN.ATM.CO2E.GF.KT',  # CO2 emissions from gaseous fuel consumption (kt)
    'EN.ATM.CO2E.LF.ZS',  # CO2 emissions from liquid fuel consumption (% of total)
    # 'EN.ATM.CO2E.LF.KT',  # CO2 emissions from liquid fuel consumption (kt)
    'EN.ATM.CO2E.SF.ZS',  # CO2 emissions from solid fuel consumption (% of total)
    # 'EN.ATM.CO2E.SF.KT',  # CO2 emissions from solid fuel consumption (kt)
    'EG.USE.ELEC.KH.PC',  # Electric power consumption (kWh per capita)
    'EG.ELC.HYRO.ZS',  # Electricity production from hydroelectric sources (% of total)
    'EG.ELC.NUCL.ZS',  # Electricity production from nuclear sources (% of total)
    'EG.ELC.FOSL.ZS',  # Electricity production from oil, gas and coal sources (% of total)
    'EG.ELC.RNWX.ZS',  # Electricity production from renewable sources, excluding hydroelectric (% of total)
    'EG.IMP.CONS.ZS',  # Energy imports, net (% of energy use)
    'EG.USE.PCAP.KG.OE',  # Energy use (kg of oil equivalent per capita)
]
columns = ['CountryCode',
           'EN.ATM.CO2E.KT',  # CO2 emissions (kt)
           'EN.ATM.CO2E.GF.ZS',  # CO2 emissions from gaseous fuel consumption (% of total)
           # 'EN.ATM.CO2E.GF.KT',  # CO2 emissions from gaseous fuel consumption (kt)
           'EN.ATM.CO2E.LF.ZS',  # CO2 emissions from liquid fuel consumption (% of total)
           # 'EN.ATM.CO2E.LF.KT',  # CO2 emissions from liquid fuel consumption (kt)
           'EN.ATM.CO2E.SF.ZS',  # CO2 emissions from solid fuel consumption (% of total)
           # 'EN.ATM.CO2E.SF.KT',  # CO2 emissions from solid fuel consumption (kt)
           'EG.USE.ELEC.KH.PC',  # Electric power consumption (kWh per capita)
           'EG.ELC.HYRO.ZS',  # Electricity production from hydroelectric sources (% of total)
           'EG.ELC.NUCL.ZS',  # Electricity production from nuclear sources (% of total)
           'EG.ELC.FOSL.ZS',  # Electricity production from oil, gas and coal sources (% of total)
           'EG.ELC.RNWX.ZS',  # Electricity production from renewable sources, excluding hydroelectric (% of total)
           'EG.IMP.CONS.ZS',  # Energy imports, net (% of energy use)
           'EG.USE.PCAP.KG.OE',  # Energy use (kg of oil equivalent per capita)
           'IncomeGroup'
           ]

indicators_subset = indicators[indicators['IndicatorCode'].isin(chosen_indicators)]
country_codes = np.unique(indicators_subset['CountryCode'])

df_new = pd.DataFrame()

for c_code in country_codes:
    # split by country code
    split_by_country = indicators_subset[indicators_subset['CountryCode'] == c_code]
    row = [c_code]

    # average of features (each)
    for feature in chosen_indicators:
        temp = split_by_country[split_by_country['IndicatorCode'] == feature]
        # if feature == 'EG.IMP.CONS.ZS':
        #     print(temp)
        if len(temp) == 0:
            row.append('None')
        else:
            row.append(temp['Value'].ewm(span=10).mean().iloc[-1])  # Exponetial Moving Average (10 years)

    # set income
    row.append(country[country['CountryCode'] == c_code]['IncomeGroup'].iloc[0])
    df_new = df_new.append([row])

df_new.columns = columns
df_new.set_index('CountryCode', drop=True, inplace=True)
df_new.fillna('Not exist', inplace=True)
df_new.to_csv('dataSet/clustring/cluster_data.csv')
