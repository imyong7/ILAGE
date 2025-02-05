import pandas as pd
import numpy as np
from decimal import Decimal
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # 데이터 가공
    data = pd.read_csv('./data/smart_info.csv')

    data = data[data['Operation Type'] == 'After Write' ]
    data.drop(columns=['ID'], axis=1, inplace=True)
    # data['DateTime'] = pd.to_datetime(data['DateTime']).astype('int64')

    data['Critical Warning'] = data['Critical Warning'].apply(lambda x: int(x, 16))
    data['Temperature'] = data['Temperature'].str.replace(' Celsius', '').astype(int)

    data['Data Units Read'] = data['Data Units Read'].str.replace(',', '').str.replace('[', '').str.replace(']', '') #.str.replace(' MB', '')
    data[['Data Units Read Numbers', 'Data Units Read Bytes', 'Read Byte Type']] = data['Data Units Read'].str.split(' ', expand=True)
    data['Data Units Read Numbers'] = data['Data Units Read Numbers'].astype(int)
    # data['Data Units Read Bytes'] = data['Data Units Read Bytes'].astype(float) / (1024 * 1024)

    data['Host Read Commands'] = data['Host Read Commands'].str.replace(',', '').astype(int)
    data['Host Write Commands'] = data['Host Write Commands'].str.replace(',', '').astype(int)
    data['Controller Busy Time'] = data['Controller Busy Time'].str.replace(',', '').astype(int)

    data['Available Spare'] = data['Available Spare'].str.replace('%', '').astype(int)
    data['Available Spare Threshold'] = data['Available Spare Threshold'].str.replace('%', '').astype(int)
    data['Percentage Used'] = data['Percentage Used'].str.replace('%', '').astype(int)

    data['Data Units Written'] = data['Data Units Written'].str.replace(',', '').str.replace('[', '').str.replace(']', '')# .str.replace(' GB', '').str.replace(' TB', '')
    data[['Data Units Written Numbers', 'Data Units Written Bytes', 'Written Byte Type']] = data['Data Units Written'].str.split(' ', expand=True)
    data['Data Units Written Numbers'] = data['Data Units Written Numbers'].astype(int)
    data['Data Units Written Bytes'] = data['Data Units Written Bytes'].astype(float)


    data.loc[data['Read Byte Type'] == 'MB', 'Data Units Read Bytes'] = data.loc[data['Read Byte Type'] == 'MB', 'Data Units Read Bytes'].astype(float) / (1024 * 1024)
    data.loc[data['Read Byte Type'] == 'GB', 'Data Units Read Bytes'] = data.loc[data['Read Byte Type'] == 'GB', 'Data Units Read Bytes'].astype(float) / 1024
    data['Data Units Read Bytes'] = data['Data Units Read Bytes'].apply(lambda x: Decimal(x).quantize(Decimal("0.0000000001")) if isinstance(x, float) else x)

    data.loc[data['Written Byte Type'] == 'GB', 'Data Units Written Bytes'] = data.loc[data['Written Byte Type'] == 'GB', 'Data Units Written Bytes'].astype(float) / 1024


    data = data[ data['Percentage Used'] < 28]

    data.drop(columns=['Operation Type', 'Critical Warning', 'Available Spare', 'Available Spare Threshold', 'Power Cycles', 'Power On Hours', 'Unsafe Shutdowns', \
                       'Media and Data Integrity Errors', 'Error Information Log Entries', 'Warning  Comp. Temperature Time', 'Critical Comp. Temperature Time', \
                       'Data Units Read', 'Data Units Written', 'Written Byte Type', 'Read Byte Type'], axis=1, inplace=True)

    data = data.reset_index(drop=True)
    data.head(5)

    data.columns = data.columns.str.replace(' ', '_')

    data.to_csv('./data/smart_info_processed.csv', float_format='%.10f', index=False)