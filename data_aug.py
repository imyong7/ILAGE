import pandas as pd
import numpy as np
from collections import Counter
pd.options.display.float_format = '{:.10f}'.format

uniform_unit = 0.1

def est_avg_datetime_diff(data):
    df = {}
    # 1. DateTime을 datetime 객체로 변환
    df['DateTime'] = pd.to_datetime(data['DateTime'])

    # 2. DateTime을 timestamp로 변환 (초 단위)
    df['Timestamp'] = df['DateTime'].astype('int64') // 10**9  # 초 단위

    # 3. 시간 차이 계산
    df['Time_Diff'] = df['Timestamp'].diff()  # 초 단위 차이 계산

    # 4. 평균 시간 차이 계산
    average_time_diff = df['Time_Diff'].mean()  # 평균 시간 차이

    return average_time_diff

def est_avg_temp_diff(data):
    df = {}
    df['Temperature'] = data['Temperature']
    # df['Temp_Avg'] = df['Temperature'].diff()  # 온도 평균 구함
    average_temp_diff = df['Temperature'].mean()  # 평균 온도 차이

    return average_temp_diff

def est_avg_host_read_cmds(data):
    df = {}
    df['Host_Read_Commands'] = data['Host_Read_Commands']
    df['Host_Read_Commands'] = df['Host_Read_Commands'].diff()
    average_hr_cmds = df['Host_Read_Commands'].mean()  # 평균 읽기

    return average_hr_cmds

def est_avg_host_write_cmds(data):
    df = {}
    df['Host_Write_Commands'] = data['Host_Write_Commands']
    df['Host_Write_Commands'] = df['Host_Write_Commands'].diff()
    average_hw_cmds = df['Host_Write_Commands'].mean()  # 평균 쓰기

    return average_hw_cmds

def est_avg_ctrl_busy_time(data):
    df = {}
    df['Controller_Busy_Time'] = data['Controller_Busy_Time']
    df['Controller_Busy_Time'] = df['Controller_Busy_Time'].diff()
    average_time = df['Controller_Busy_Time'].mean()  # 평균 시간

    return average_time

def est_avg_du_read_numbers(data):
    df = {}
    df['Data_Units_Read_Numbers'] = data['Data_Units_Read_Numbers']
    # df['Data_Units_Read_Numbers'] = df['Data_Units_Read_Numbers'].diff()

    nums = {}
    for num in list(df['Data_Units_Read_Numbers']):
        if num in nums:
            nums[num] = nums[num] + 1
        else:
            nums[num] = 1

    sum_avg_read_num = 0
    for key in nums:
        count = nums[key]
        sum_avg_read_num = sum_avg_read_num + (1 / count)

    sum_avg_read_num = sum_avg_read_num / len(nums)

    return sum_avg_read_num

def est_avg_du_read_bytes(data):
    df = {}
    df['Data_Units_Read_Bytes'] = data['Data_Units_Read_Bytes']
    # average_bytes = df['Data_Units_Read_Bytes'].mean()  # 평균 읽기 bytes

    # return average_bytes

    bytes = {}
    for byte in list(df['Data_Units_Read_Bytes']):
        if byte in bytes:
            bytes[byte] = bytes[byte] + 1
        else:
            bytes[byte] = 1

    sum_avg_read_byte = 0
    for key in bytes:
        count = bytes[key]
        sum_avg_read_byte = sum_avg_read_byte + (1 / count)

    sum_avg_read_byte = sum_avg_read_byte / len(bytes)
    return sum_avg_read_byte

def est_avg_du_write_numbers(data):
    df = {}
    df['Data_Units_Written_Numbers'] = data['Data_Units_Written_Numbers']
    average_num = df['Data_Units_Written_Numbers'].mean()  # 평균 쓰기 숫자

    return average_num

def est_avg_du_write_bytes(data):
    df = {}
    df['Data_Units_Written_Bytes'] = data['Data_Units_Written_Bytes']
    # average_bytes = df['Data_Units_Written_Bytes'].mean()  # 평균 쓰기 bytes
    # 
    # return average_bytes

    bytes = {}
    for byte in list(df['Data_Units_Written_Bytes']):
        if byte in bytes:
            bytes[byte] = bytes[byte] + 1
        else:
            bytes[byte] = 1

    sum_avg_write_byte = 0
    for key in bytes:
        count = bytes[key]
        sum_avg_write_byte = sum_avg_write_byte + (1 / count)

    sum_avg_write_byte = sum_avg_write_byte / len(bytes)
    return sum_avg_write_byte


if __name__ == "__main__":
    data = pd.read_csv('./data/smart_info_processed.csv')
    data = data.reset_index(drop=True)


    latest_percentage_used = int(data.iloc[-1]['Percentage_Used'])
    sum_per_used = 0

    start_per_used = 10

    for i in range(latest_percentage_used + 1):
        len_rows = len(data[ data['Percentage_Used'] == i ])
        sum_per_used += len_rows

    # 1. Percentage Used의 평균을 구함, 100%가 되면 SSD를 교체한다. - Loop 돌릴 조건이 됨
    avg_per_used = int(sum_per_used / (latest_percentage_used + 1))
    print(f"평균 Percentage Used : {avg_per_used}")

    # 정규분포를 통해 loop 돌릴 횟수를 결정함
    unif_avg_per_used = int(avg_per_used + avg_per_used * np.random.uniform(-1 * uniform_unit, uniform_unit))
    print(f"Loop를 돌릴 횟수 결정 : {unif_avg_per_used}")


    df_last_clone = data.iloc[-1].copy(deep=True)

    # print(df_last_clone)

    # 2. 데이터가 발생한 평균 시간을 구한다.
    avg_datetime_diff = est_avg_datetime_diff(data)
    unif_datetime_diff = int(avg_datetime_diff + avg_datetime_diff * np.random.uniform(-1 * uniform_unit, uniform_unit))
    print(f"평균/정규 시간 차이: {avg_datetime_diff}, {unif_datetime_diff}")

    # 3. 평균 Temperature의 증감율을 구한다.
    avg_temp_diff = int(est_avg_temp_diff(data))
    unif_temp_diff = int(avg_temp_diff + avg_temp_diff * np.random.uniform(-1 * uniform_unit / 2, uniform_unit / 2))
    print(f"평균/정규 온도 차이: {avg_temp_diff}, {unif_temp_diff}")

    # 4. 평균 Host Read Commands 증감율을 구한다.
    avg_host_read_cmds = int(est_avg_host_read_cmds(data))
    unif_host_read_cmds = int(avg_host_read_cmds + avg_host_read_cmds * np.random.uniform(0, uniform_unit * 2))
    print(f"평균/정규 Host Read Cmds 차이: {avg_host_read_cmds}, {unif_host_read_cmds}")

    # 5. 평균 Host Write Commands 증감율을 구한다.
    avg_host_write_cmds = int(est_avg_host_write_cmds(data))
    unif_host_write_cmds = int(avg_host_write_cmds + avg_host_write_cmds * np.random.uniform(0, uniform_unit))
    print(f"평균/정규 Host Write Cmds 차이: {avg_host_write_cmds}, {unif_host_write_cmds}")

    # 6. 평균 Ctrl. Busy Time 증감율을 구한다.
    avg_ctrl_busy_time = est_avg_ctrl_busy_time(data)
    unif_ctrl_busy_time = round(avg_ctrl_busy_time + avg_ctrl_busy_time * np.random.uniform(0, uniform_unit / 2), 2)
    print(f"평균/정규 Ctrl. Busy Time 차이: {avg_ctrl_busy_time}, {unif_ctrl_busy_time}")

    # 7. 평균 Data Units Read Numbers 증감율을 구한다.
    avg_du_read_numbers = est_avg_du_read_numbers(data)
    unif_du_read_numbers = round(avg_du_read_numbers + avg_du_read_numbers * np.random.uniform(0, uniform_unit), 2)
    print(f"평균/정규 Data Units Read Numbers 차이: {avg_du_read_numbers}, {unif_du_read_numbers}")

    # 8. 평균 Data Units Read Bytes 증감율을 구한다.
    avg_du_read_bytes = est_avg_du_read_bytes(data)
    unif_du_read_bytes = round(avg_du_read_bytes + avg_du_read_bytes * np.random.uniform(-1 * uniform_unit, uniform_unit), 2)
    print(f"평균/정규 Data Units Read Bytes 차이: {avg_du_read_bytes}, {unif_du_read_bytes}")

    # 9. 평균 Data Units Write Numbers 증감율을 구한다.
    avg_du_write_numbers = int(est_avg_du_write_numbers(data))
    unif_du_write_numbers = int(avg_du_write_numbers + avg_du_write_numbers * np.random.uniform(0, uniform_unit / 2))
    print(f"평균/정규 Data Units Write Numbers 차이: {avg_du_write_numbers}, {unif_du_write_numbers}")

    # 10. 평균 Data Units Write Bytes 증감율을 구한다.
    avg_du_write_bytes = est_avg_du_write_bytes(data)
    unif_du_write_bytes = round(avg_du_write_bytes + avg_du_write_bytes * np.random.uniform(0, uniform_unit / 2), 2)
    print(f"평균/정규 Data Units Write Bytes 차이: {avg_du_write_bytes}, {unif_du_write_bytes}")


    df_last_clone['DateTime'] = str(pd.to_datetime(df_last_clone['DateTime']) + pd.Timedelta(seconds=unif_datetime_diff))
    df_last_clone['Temperature'] = unif_temp_diff
    df_last_clone['Percentage_Used'] = 10
    df_last_clone['Host_Read_Commands'] = df_last_clone['Host_Read_Commands'] + unif_host_read_cmds
    df_last_clone['Host_Write_Commands'] = df_last_clone['Host_Write_Commands'] + unif_host_write_cmds
    df_last_clone['Controller_Busy_Time'] = df_last_clone['Controller_Busy_Time'] + unif_ctrl_busy_time
    df_last_clone['Data_Units_Read_Numbers'] = df_last_clone['Data_Units_Read_Numbers'] + unif_du_read_numbers
    df_last_clone['Data_Units_Read_Bytes'] = df_last_clone['Data_Units_Read_Bytes'] + unif_du_read_bytes
    df_last_clone['Data_Units_Written_Numbers'] = df_last_clone['Data_Units_Written_Numbers'] + unif_du_write_numbers
    df_last_clone['Data_Units_Written_Bytes'] = df_last_clone['Data_Units_Written_Bytes'] + unif_du_write_bytes

    data.loc[len(data)] = df_last_clone
    # data = pd.concat([data, df_last_clone], ignore_index=True)

    # print(df_last_clone)
    # print(data)

    # tot_count = 0
    for start in range(start_per_used, 100):
        print(f"Percentage Used :: {start}")

        for iter in range(unif_avg_per_used):
            # print(start, iter, unif_avg_per_used)
            df_last_clone = data.iloc[-1].copy(deep=True)
            # print(df_last_clone)

            # 2. 데이터가 발생한 평균 시간을 구한다.
            avg_datetime_diff = est_avg_datetime_diff(data)
            unif_datetime_diff = int(avg_datetime_diff + avg_datetime_diff * np.random.uniform(-1 * uniform_unit, uniform_unit))
            # print(f"평균/정규 시간 차이: {avg_datetime_diff}, {unif_datetime_diff}")

            # 3. 평균 Temperature의 증감율을 구한다.
            avg_temp_diff = int(est_avg_temp_diff(data))
            unif_temp_diff = int(avg_temp_diff + avg_temp_diff * np.random.uniform(-1 * uniform_unit / 2, uniform_unit / 2))
            # print(f"평균/정규 온도 차이: {avg_temp_diff}, {unif_temp_diff}")

            # 4. 평균 Host Read Commands 증감율을 구한다.
            avg_host_read_cmds = int(est_avg_host_read_cmds(data))
            unif_host_read_cmds = int(avg_host_read_cmds + avg_host_read_cmds * np.random.uniform(0, uniform_unit * 2))
            # print(f"평균/정규 Host Read Cmds 차이: {avg_host_read_cmds}, {unif_host_read_cmds}")

            # 5. 평균 Host Write Commands 증감율을 구한다.
            avg_host_write_cmds = int(est_avg_host_write_cmds(data))
            unif_host_write_cmds = int(avg_host_write_cmds + avg_host_write_cmds * np.random.uniform(0, uniform_unit))
            # print(f"평균/정규 Host Write Cmds 차이: {avg_host_write_cmds}, {unif_host_write_cmds}")

            # 6. 평균 Ctrl. Busy Time 증감율을 구한다.
            avg_ctrl_busy_time = est_avg_ctrl_busy_time(data)
            unif_ctrl_busy_time = round(avg_ctrl_busy_time + avg_ctrl_busy_time * np.random.uniform(0, uniform_unit / 2), 2)
            # print(f"평균/정규 Ctrl. Busy Time 차이: {avg_ctrl_busy_time}, {unif_ctrl_busy_time}")

            # 7. 평균 Data Units Read Numbers 증감율을 구한다.
            avg_du_read_numbers = est_avg_du_read_numbers(data)
            unif_du_read_numbers = round(avg_du_read_numbers + avg_du_read_numbers * np.random.uniform(0, uniform_unit), 2)
            # print(f"평균/정규 Data Units Read Numbers 차이: {avg_du_read_numbers}, {unif_du_read_numbers}")

            # 8. 평균 Data Units Read Bytes 증감율을 구한다.
            avg_du_read_bytes = est_avg_du_read_bytes(data)
            unif_du_read_bytes = round(avg_du_read_bytes + avg_du_read_bytes * np.random.uniform(-1 * uniform_unit, uniform_unit), 2)
            # print(f"평균/정규 Data Units Read Bytes 차이: {avg_du_read_bytes}, {unif_du_read_bytes}")

            # 9. 평균 Data Units Write Numbers 증감율을 구한다.
            avg_du_write_numbers = int(est_avg_du_write_numbers(data))
            unif_du_write_numbers = int(avg_du_write_numbers + avg_du_write_numbers * np.random.uniform(0, uniform_unit / 2))
            # print(f"평균/정규 Data Units Write Numbers 차이: {avg_du_write_numbers}, {unif_du_write_numbers}")

            # 10. 평균 Data Units Write Bytes 증감율을 구한다.
            avg_du_write_bytes = est_avg_du_write_bytes(data)
            unif_du_write_bytes = round(avg_du_write_bytes + avg_du_write_bytes * np.random.uniform(0, uniform_unit / 2), 2)
            # print(f"평균/정규 Data Units Write Bytes 차이: {avg_du_write_bytes}, {unif_du_write_bytes}")


            df_last_clone['DateTime'] = str(pd.to_datetime(df_last_clone['DateTime']) + pd.Timedelta(seconds=unif_datetime_diff))
            df_last_clone['Temperature'] = unif_temp_diff
            df_last_clone['Percentage_Used'] = start
            df_last_clone['Host_Read_Commands'] = df_last_clone['Host_Read_Commands'] + unif_host_read_cmds
            df_last_clone['Host_Write_Commands'] = df_last_clone['Host_Write_Commands'] + unif_host_write_cmds
            df_last_clone['Controller_Busy_Time'] = df_last_clone['Controller_Busy_Time'] + unif_ctrl_busy_time
            df_last_clone['Data_Units_Read_Numbers'] = df_last_clone['Data_Units_Read_Numbers'] + unif_du_read_numbers
            df_last_clone['Data_Units_Read_Bytes'] = df_last_clone['Data_Units_Read_Bytes'] + unif_du_read_bytes
            df_last_clone['Data_Units_Written_Numbers'] = abs(df_last_clone['Data_Units_Written_Numbers'] + unif_du_write_numbers)
            df_last_clone['Data_Units_Written_Bytes'] = df_last_clone['Data_Units_Written_Bytes'] + unif_du_write_bytes

            data.loc[len(data)] = df_last_clone

        unif_avg_per_used = int(unif_avg_per_used + unif_avg_per_used * np.random.uniform(-1 * uniform_unit / 7, uniform_unit / 7))

    data.to_csv("./smart_info_data_augmented.csv", float_format='%.10f', index=False)
    print(data)




