import numpy as np

from herbie import Herbie, FastHerbie
from herbie import FastHerbie

import xarray as xr
import pandas as pd
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import time
from pathlib import Path

def get_tcc_fcasts_fastherbie(start, end, freq, fxx_range, plants_df,
                              region_extent=[0,359.999,-90,90], num_members=30,
                              save=True, save_dir='', attempts=5, remove_grib=False):
    # list of GEFS ensemble members, e.g., 'p01', 'p02', etc.
    member_list = [f"p{x:02d}" for x in range(1, num_members+1)]

    # range of initialization dates/times to pull forecasts for
    DATES = pd.date_range(start=start, end=end, freq=freq)

    # filename for saving a netcdf file with xarray dataset results
    filename = ('gefs_total_cloud_cover_f' + str(min(fxx_range)) + '-f' + 
                str(max(fxx_range)) + '_' + str(fxx_range.step) + 'h_' + 
                DATES[0].strftime('%Y%m%d%H') + '-' + 
                DATES[-1].strftime('%Y%m%d%H') + '_' + freq + '_step_' + 
                str(num_members) + '_members')
    the_file = Path(save_dir + filename + '.nc')
    if the_file.is_file():
        # try to load dataset from existing netcdf file
        # print('try loading ' + filename + '.nc')
        ds = xr.open_dataset(save_dir + filename + '.nc')
        print('loaded ' + filename + '.nc')
    else:
        ds_dict={}
        print('downloading new data')
        for x in range(0, num_members):
            for attempt in range(attempts):
                try:
                    if attempt==0:
                        # try downloading
                        ds_dict[x] = FastHerbie(DATES, model="gefs", product="atmos.5",
                                        member=member_list[x], fxx=fxx_range).xarray("TCDC",
                                                                            remove_grib=remove_grib,
                                                                            )
                    else:
                        # after first attempt, set overwrite=True to overwrite partial files
                        ds_dict[x] = FastHerbie(DATES, model="gefs", product="atmos.5",
                                        member=member_list[x], fxx=fxx_range).xarray("TCDC",
                                                                            remove_grib=remove_grib,
                                                                            overwrite=True
                                                                            )
                except:
                    print('attempt ' + str(attempt+1) + ', pausing for ' + str((attempt+1)**2) + ' min')
                    time.sleep(60*(attempt+1)**2)
                else:
                    break
            else:
                raise ValueError('download failed, ran out of attempts')
                # return 'ERROR'
            # lat, lon extent for region of interest in GEFS coordinates
            # longitude has to be between 0 and 360
            min_lon = (region_extent[0]+360) % 360
            max_lon = (region_extent[1]+360) % 360
            min_lat = region_extent[2]
            max_lat = region_extent[3]

            # slice to region extent, add member name to tcc, drop unneeded coordinates
            # note that lat is sliced max to min because of order that latitude is stored in with GEFS
            # (I think...)
            ds_dict[x] = ds_dict[x].sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon,max_lon))
            ds_dict[x] = ds_dict[x].rename({'tcc':'tcc_' + member_list[x]})
            ds_dict[x] = ds_dict[x].drop_vars(['number', 'atmosphere', 'gribfile_projection'])
            print('member ' + str(x+1) + ' download complete')

        # merge datasets
        ds = xr.merge([ds_dict[i] for i in ds_dict])

        del ds_dict
        print('downloaded and processed ')
        if save==True:
            # save dataset as netcdf file
            # added encoding to fix issue with timestamps (?) being messed up
            # also reduces file size by about 50-80%
            comp = dict(zlib=True, complevel=1)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(save_dir + filename + '.nc', encoding=encoding)
            print('saved ' + filename + '.nc')

    # GEFS uses longitudes in [0,360), so use modulo to convert lon to be 0 <= lon < 360
    lon_gefs = [(x + 360) % 360 for x in plants_df.longitude]

    # select the whole list of sites at once
    lats = xr.DataArray(plants_df.latitude.values, dims='plant_number') 
    lons = xr.DataArray(lon_gefs, dims='plant_number')
    df = ds.sel(latitude = lats, longitude = lons, method = 'nearest').to_dataframe()

    # assign ac capacity based on plant_number and index in plants_df
    df = df.merge(plants_df[['ac_capacity']], left_on='plant_number', right_index=True)

    # remove index
    df = df.reset_index()

    # rename latitude, longitude columns to note that they came from the GEFS coordinates
    # rename time to add utc
    df = df.rename(columns={'latitude':'gefs_latitude','longitude':'gefs_longitude',
                            'time':'time_utc'
                })

    # loop through unique forecasts (unique 'time_utc' and 'step' values)
    temp = {}
    i=0
    for time_utc in df['time_utc'].unique():
        for step in df['step'].unique():
            
            df_temp = df[(df['time_utc']==time_utc) & (df['step']==step)]
            out = df_temp.filter(regex='tcc').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            temp[i] = out.to_frame().T
            temp[i].insert(0, 'time_utc', time_utc)
            temp[i].insert(1, 'step', step)
            # because each time, step combination has a unique valid_time:
            valid_time = df_temp['valid_time'].unique()[0]
            # add valid_time
            temp[i].insert(2, 'valid_time', valid_time)
            i += 1

    # concatenate results
    weighted_avg_tcc = pd.concat(temp)
    del temp, df_temp, df

    # rename columns, set indices
    weighted_avg_tcc.rename(columns={'valid_time':'valid_time_utc_end_of_interval'},
                                    inplace=True)
    weighted_avg_tcc.set_index(['time_utc', 'step','valid_time_utc_end_of_interval'],inplace=True)

    # calculate stdev of weighted avg tcc for each valid_time_utc
    weighted_avg_tcc.insert(0,'tcc_std',weighted_avg_tcc.filter(regex='tcc').std(axis=1))

    weighted_avg_tcc = weighted_avg_tcc.reset_index().set_index('valid_time_utc_end_of_interval')
    temp1 = weighted_avg_tcc.copy()
    temp2 = weighted_avg_tcc.copy()
    temp1.index = weighted_avg_tcc.index.shift(-1, freq='h')
    temp2.index = weighted_avg_tcc.index.shift(-2, freq='h')
    weighted_avg_tcc = pd.concat([weighted_avg_tcc, temp1, temp2])
    del temp1, temp2

    weighted_avg_tcc = weighted_avg_tcc.sort_index()

    ds.close()
    return weighted_avg_tcc

def get_hml_tcc_fcasts_fastherbie(start, end, freq, fxx_range, plants_df,
                              region_extent=[0,359.999,-90,90], num_members=30,
                              save=True, save_dir='', attempts=5, remove_grib=False):
    # list of GEFS ensemble members, e.g., 'p01', 'p02', etc.
    member_list = [f"p{x:02d}" for x in range(1, num_members+1)]

    # range of initialization dates/times to pull forecasts for
    DATES = pd.date_range(start=start, end=end, freq=freq)

    # filename for saving a netcdf file with xarray dataset results
    filename = ('gefs_hml_cloud_cover_f' + str(min(fxx_range)) + '-f' + 
                str(max(fxx_range)) + '_' + str(fxx_range.step) + 'h_' + 
                DATES[0].strftime('%Y%m%d%H') + '-' + 
                DATES[-1].strftime('%Y%m%d%H') + '_' + freq + '_step_' + 
                str(num_members) + '_members')
    the_file = Path(save_dir + filename + '.nc')
    if the_file.is_file():
        # try to load dataset from existing netcdf file
        # print('try loading ' + filename + '.nc')
        ds = xr.open_dataset(save_dir + filename + '.nc')
        print('loaded ' + filename + '.nc')
    else:
        ds_dict={}
        print('downloading new data')
        for x in range(0, num_members):
            for attempt in range(attempts):
                try:
                    if attempt==0:
                        # try downloading
                        ds_dict[x] = FastHerbie(DATES, model="gefs", product="atmos.5b",
                                        member=member_list[x], fxx=fxx_range).xarray(":TCDC:[low|middle|high]",
                                                                            remove_grib=remove_grib,
                                                                            )
                    else:
                        # after first attempt, set overwrite=True to overwrite partial files
                        ds_dict[x] = FastHerbie(DATES, model="gefs", product="atmos.5b",
                                        member=member_list[x], fxx=fxx_range).xarray(":TCDC:[low|middle|high]",
                                                                            remove_grib=remove_grib,
                                                                            overwrite=True
                                                                            )
                except:
                    print('attempt ' + str(attempt+1) + ', pausing for ' + str((attempt+1)**2) + ' min')
                    time.sleep(60*(attempt+1)**2)
                else:
                    break
            else:
                raise ValueError('download failed, ran out of attempts')
                # return 'ERROR'
            # lat, lon extent for region of interest in GEFS coordinates
            # longitude has to be between 0 and 360
            min_lon = (region_extent[0]+360) % 360
            max_lon = (region_extent[1]+360) % 360
            min_lat = region_extent[2]
            max_lat = region_extent[3]

            # pull out high, low, and middle cloud cover
            ds_dict[x] = ds_dict[x].assign(tcc_high=ds_dict[x].isel(step=0).tcc)
            ds_dict[x] = ds_dict[x].assign(tcc_low=ds_dict[x].isel(step=1).tcc)
            ds_dict[x] = ds_dict[x].assign(tcc_middle=ds_dict[x].isel(step=2).tcc)
            # ds_dict[x] = ds_dict[x].assign(tcc_sum_low_mid=ds_dict[x].isel(step=1).tcc + ds_dict[x].isel(step=2).tcc)

            # slice to region extent, add member name to tcc, drop unneeded coordinates
            # note that lat is sliced max to min because of order that latitude is stored in with GEFS
            # (I think...)
            ds_dict[x] = ds_dict[x].sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon,max_lon))
            
            # ds_dict[x] = ds_dict[x].rename({'tcc':'tcc_' + member_list[x]})
            # add member name to variable names
            ds_dict[x] = ds_dict[x].rename({'tcc_high':'tcc_high_' + member_list[x]})
            ds_dict[x] = ds_dict[x].rename({'tcc_low':'tcc_low_' + member_list[x]})
            ds_dict[x] = ds_dict[x].rename({'tcc_middle':'tcc_middle_' + member_list[x]})
            # ds_dict[x] = ds_dict[x].rename({'tcc_sum_low_mid':'tcc_sum_low_mid_' + member_list[x]})

            # drop variables that are not needed
            ds_dict[x] = ds_dict[x].drop_vars(['tcc', 'number', 'gribfile_projection', 'highCloudLayer',
                            'lowCloudLayer', 'middleCloudLayer'])
            print('member ' + str(x+1) + ' download complete')

        # merge datasets
        ds = xr.merge([ds_dict[i] for i in ds_dict])

        del ds_dict
        print('downloaded and processed ')
        if save==True:
            # save dataset as netcdf file
            # added encoding to fix issue with timestamps (?) being messed up
            # also reduces file size by about 50-80%
            comp = dict(zlib=True, complevel=1)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(save_dir + filename + '.nc', encoding=encoding)
            print('saved ' + filename + '.nc')

    # GEFS uses longitudes in [0,360), so use modulo to convert lon to be 0 <= lon < 360
    lon_gefs = [(x + 360) % 360 for x in plants_df.longitude]

    # select the whole list of sites at once
    lats = xr.DataArray(plants_df.latitude.values, dims='plant_number') 
    lons = xr.DataArray(lon_gefs, dims='plant_number')
    df = ds.sel(latitude = lats, longitude = lons, method = 'nearest').to_dataframe()

    # assign ac capacity based on plant_number and index in plants_df
    df = df.merge(plants_df[['ac_capacity']], left_on='plant_number', right_index=True)

    # remove index
    df = df.reset_index()

    # drop duplicates
    df = df.drop_duplicates()

    # rename latitude, longitude columns to note that they came from the GEFS coordinates
    # rename time to add utc
    df = df.rename(columns={'latitude':'gefs_latitude','longitude':'gefs_longitude',
                            'time':'time_utc'
                })

    # loop through members and calculate sum of low and middle cloud cover
    for x in range(0, num_members):
        df['tcc_sum_low_mid_' + member_list[x]] = df['tcc_middle_' + member_list[x]] + df['tcc_low_' + member_list[x]]

    # loop through unique forecasts (unique 'time_utc' and 'step' values)
    temp = {}
    i=0
    for time_utc in df['time_utc'].unique():
        for step in df['step'].unique():
            
            df_temp = df[(df['time_utc']==time_utc) & (df['step']==step)]
            # out = df_temp.filter(regex='tcc').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out_low = df_temp.filter(regex='tcc_low').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out_high = df_temp.filter(regex='tcc_high').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out_mid = df_temp.filter(regex='tcc_mid').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            # out_sum_low_mid = df_temp.filter(regex='tcc_mid|tcc_low').sum(axis=1).multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out_sum_low_mid = df_temp.filter(regex='tcc_sum_low_mid').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out = pd.concat([out_low, out_high, out_mid, out_sum_low_mid])
            temp[i] = out.to_frame().T
            temp[i].insert(0, 'time_utc', time_utc)
            temp[i].insert(1, 'step', step)
            # because each time, step combination has a unique valid_time:
            valid_time = df_temp['valid_time'].unique()[0]
            # add valid_time
            temp[i].insert(2, 'valid_time', valid_time)
            i += 1

    # concatenate results
    weighted_avg_tcc = pd.concat(temp)
    del temp, df_temp, df

    # rename columns, set indices
    weighted_avg_tcc.rename(columns={'valid_time':'valid_time_utc_end_of_interval'},
                                    inplace=True)
    weighted_avg_tcc.set_index(['time_utc', 'step','valid_time_utc_end_of_interval'],inplace=True)

    # calculate stdev of weighted avg tcc for each valid_time_utc
    # weighted_avg_tcc.insert(0,'tcc_std',weighted_avg_tcc.filter(regex='tcc').std(axis=1))
    weighted_avg_tcc.insert(0,'tcc_low_std',weighted_avg_tcc.filter(regex='tcc_low').std(axis=1))
    weighted_avg_tcc.insert(0,'tcc_high_std',weighted_avg_tcc.filter(regex='tcc_high').std(axis=1))
    weighted_avg_tcc.insert(0,'tcc_mid_std',weighted_avg_tcc.filter(regex='tcc_mid').std(axis=1))
    weighted_avg_tcc.insert(0,'tcc_sum_low_mid_std',weighted_avg_tcc.filter(regex='tcc_sum_low_mid').std(axis=1))

    weighted_avg_tcc = weighted_avg_tcc.reset_index().set_index('valid_time_utc_end_of_interval')
    temp1 = weighted_avg_tcc.copy()
    temp2 = weighted_avg_tcc.copy()
    temp1.index = weighted_avg_tcc.index.shift(-1, freq='h')
    temp2.index = weighted_avg_tcc.index.shift(-2, freq='h')
    weighted_avg_tcc = pd.concat([weighted_avg_tcc, temp1, temp2])
    del temp1, temp2

    weighted_avg_tcc = weighted_avg_tcc.sort_index()

    ds.close()
    return weighted_avg_tcc

def get_many_tcc_fcasts_fastherbie(start, end, freq, query_len_days, fxx_range,
                                   plants_df, region_extent=[0,359.999,-90,90],
                                   num_members=30, save=True, save_dir='',
                                   attempts=5,remove_grib=False):
    end_dt = pd.to_datetime(end)
    current_start_dt = pd.to_datetime(start)
    data_dict = {} 
    i=0
    while current_start_dt <= end_dt:
        # define start and end time
        current_start = current_start_dt.strftime('%Y-%m-%dT%H:%M:%S')
        current_end_dt = current_start_dt + relativedelta(days=query_len_days) - pd.Timedelta(freq)
        current_end = current_end_dt.strftime('%Y-%m-%dT%H:%M:%S')
        # get data
        data_dict[i] = get_tcc_fcasts_fastherbie(current_start, current_end,
                                                freq, fxx_range,
                                                plants_df, num_members=num_members,
                                                region_extent=region_extent,
                                                save=save, save_dir=save_dir,
                                                attempts=attempts,
                                                remove_grib=remove_grib)

        # increment start datetime and index
        current_start_dt = current_end_dt + pd.Timedelta(freq)
        i += 1

    weighted_avg_tcc_all = pd.concat(data_dict)
    return weighted_avg_tcc_all

def get_many_hml_tcc_fcasts_fastherbie(start, end, freq, query_len_days, fxx_range,
                                   plants_df, region_extent=[0,359.999,-90,90],
                                   num_members=30, save=True, save_dir='',
                                   attempts=5,remove_grib=False):
    end_dt = pd.to_datetime(end)
    current_start_dt = pd.to_datetime(start)
    data_dict = {} 
    i=0
    while current_start_dt <= end_dt:
        # define start and end time
        current_start = current_start_dt.strftime('%Y-%m-%dT%H:%M:%S')
        current_end_dt = current_start_dt + relativedelta(days=query_len_days) - pd.Timedelta(freq)
        current_end = current_end_dt.strftime('%Y-%m-%dT%H:%M:%S')
        # get data
        data_dict[i] = get_hml_tcc_fcasts_fastherbie(current_start, current_end,
                                                freq, fxx_range,
                                                plants_df, num_members=num_members,
                                                region_extent=region_extent,
                                                save=save, save_dir=save_dir,
                                                attempts=attempts,
                                                remove_grib=remove_grib)

        # increment start datetime and index
        current_start_dt = current_end_dt + pd.Timedelta(freq)
        i += 1

    weighted_avg_tcc_all = pd.concat(data_dict)
    return weighted_avg_tcc_all

def get_hml_tcc_fcast(init_time,fxx,plants_df,num_members=30,
                   verbose=False,remove_grib=False,overwrite=False,
                   region_extent=[0,359.999,-90,90]):
    # list of GEFS ensemble members, e.g., 'p01', 'p02', etc.
    member_list = [f"p{x:02d}" for x in range(1, num_members+1)]

    # GEFS uses longitudes in [0,360), so use modulo to convert lon to be 0 <= lon < 360
    lon_gefs = [(x + 360) % 360 for x in plants_df.longitude]

    d={}
    data=[]
    member_site_tcc = np.empty([num_members, len(plants_df.index)])
    for x in range(0, num_members):
        d[x] = Herbie(init_time, model="gefs", product="atmos.5b", 
                            member=member_list[x], fxx=fxx,
                            verbose=verbose).xarray(":TCDC:[low|middle|high]",
                                                        remove_grib=remove_grib,
                                                        overwrite=overwrite)

        # tcc is a 3d variable; pull out high, low, and middle cloud cover
        # d[x] = d[x].assign(tcc_high=d[x].tcc[0])
        # d[x] = d[x].assign(tcc_low=d[x].tcc[1])
        # d[x] = d[x].assign(tcc_middle=d[x].tcc[2])

        # alternative for same result:
        # this returns a dataset with 3 repeated "step" coordinates; 
        # pull out high, low, and middle cloud cover
        d[x] = d[x].assign(tcc_high=d[x].isel(step=0).tcc)
        d[x] = d[x].assign(tcc_low=d[x].isel(step=1).tcc)
        d[x] = d[x].assign(tcc_middle=d[x].isel(step=2).tcc)

        # add member name to variable names
        d[x] = d[x].rename({'tcc_high':'tcc_high_' + member_list[x]})
        d[x] = d[x].rename({'tcc_low':'tcc_low_' + member_list[x]})
        d[x] = d[x].rename({'tcc_middle':'tcc_middle_' + member_list[x]})

        # drop variables that are not needed
        d[x] = d[x].drop_vars(['tcc', 'number', 'gribfile_projection', 'highCloudLayer',
                            'lowCloudLayer', 'middleCloudLayer'])

    ds = xr.merge([d[i] for i in d])

    # GEFS uses longitudes in [0,360), so use modulo to convert lon to be 0 <= lon < 360
    lon_gefs = [(x + 360) % 360 for x in plants_df.longitude]

    # select the whole list of sites at once
    lats = xr.DataArray(plants_df.latitude.values, dims='plant_number') 
    lons = xr.DataArray(lon_gefs, dims='plant_number')
    df = ds.isel(step=0).sel(latitude = lats, longitude = lons, method = 'nearest').to_dataframe() # <-- just get first step, since steps are repeated across variables


    # assign ac capacity based on plant_number and index in plants_df
    df = df.merge(plants_df[['ac_capacity']], left_on='plant_number', right_index=True)

    # rename latitude, longitude columns to note that they came from the GEFS coordinates
    # rename time to add utc
    df = df.rename(columns={'latitude':'gefs_latitude','longitude':'gefs_longitude',
                            'time':'time_utc'
                })

    # remove index
    df = df.reset_index()

    # loop through unique forecasts (unique 'time_utc' and 'step' values)
    temp = {}
    i=0
    for time_utc in df['time_utc'].unique():
        for step in df['step'].unique():
            
            df_temp = df[(df['time_utc']==time_utc) & (df['step']==step)]
            out_low = df_temp.filter(regex='tcc_low').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out_high = df_temp.filter(regex='tcc_high').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out_mid = df_temp.filter(regex='tcc_mid').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            out = pd.concat([out_low, out_high, out_mid])
            temp[i] = out.to_frame().T
            temp[i].insert(0, 'time_utc', time_utc)
            temp[i].insert(1, 'step', step)
            # because each time, step combination has a unique valid_time:
            valid_time = df_temp['valid_time'].unique()[0]
            # add valid_time
            temp[i].insert(2, 'valid_time', valid_time)
            i += 1

    df_temp = df[(df['time_utc']==time_utc) & (df['step']==step)]

    # concatenate results
    weighted_avg_tcc = pd.concat(temp)
    del temp, df_temp, df

    # rename columns, set indices
    weighted_avg_tcc.rename(columns={'valid_time':'valid_time_utc_end_of_interval'},
                                    inplace=True)
    weighted_avg_tcc.set_index(['time_utc', 'step','valid_time_utc_end_of_interval'],inplace=True)

    # calculate stdev of weighted avg tcc for each valid_time_utc
    # weighted_avg_tcc.insert(0,'tcc_std',weighted_avg_tcc.filter(regex='tcc').std(axis=1))
    weighted_avg_tcc.insert(0,'tcc_low_std',weighted_avg_tcc.filter(regex='tcc_low').std(axis=1))
    weighted_avg_tcc.insert(0,'tcc_high_std',weighted_avg_tcc.filter(regex='tcc_high').std(axis=1))
    weighted_avg_tcc.insert(0,'tcc_mid_std',weighted_avg_tcc.filter(regex='tcc_mid').std(axis=1))

    weighted_avg_tcc = weighted_avg_tcc.reset_index().set_index('valid_time_utc_end_of_interval')

    # slice to region of interest
    # lat, lon extent for region of interest in GEFS coordinates
    # longitude has to be between 0 and 360
    min_lon = (region_extent[0]+360) % 360
    max_lon = (region_extent[1]+360) % 360
    min_lat = region_extent[2]
    max_lat = region_extent[3]

    # slice to region extent
    # note that lat is sliced max to min because of order that latitude is stored in with GEFS
    # (I think...)
    ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon,max_lon))
    # ds = ds.rename({'tcc':'tcc_' + member_list})
    # ds = ds.drop_vars(['number', 'atmosphere', 'gribfile_projection'])

    ds.close()
    return weighted_avg_tcc, ds

def get_tcc_fcast(init_time,fxx,plants_df,num_members=30,
                   verbose=False,remove_grib=False,overwrite=False,
                   region_extent=[0,359.999,-90,90]):
    # list of GEFS ensemble members, e.g., 'p01', 'p02', etc.
    member_list = [f"p{x:02d}" for x in range(1, num_members+1)]

    # GEFS uses longitudes in [0,360), so use modulo to convert lon to be 0 <= lon < 360
    lon_gefs = [(x + 360) % 360 for x in plants_df.longitude]

    d={}
    data=[]
    member_site_tcc = np.empty([num_members, len(plants_df.index)])
    for x in range(0, num_members):
        d[x] = Herbie(init_time, model="gefs", product="atmos.5", 
                            member=member_list[x], fxx=fxx,
                            verbose=verbose).xarray("TCDC",
                                                     remove_grib=remove_grib,
                                                     overwrite=overwrite)
        d[x] = d[x].rename({'tcc':'tcc_' + member_list[x]})
        d[x] = d[x].drop_vars(['number', 'atmosphere', 'gribfile_projection'])
    ds = xr.merge([d[i] for i in d])

    # GEFS uses longitudes in [0,360), so use modulo to convert lon to be 0 <= lon < 360
    lon_gefs = [(x + 360) % 360 for x in plants_df.longitude]

    # select the whole list of sites at once
    lats = xr.DataArray(plants_df.latitude.values, dims='plant_number') 
    lons = xr.DataArray(lon_gefs, dims='plant_number')
    df = ds.sel(latitude = lats, longitude = lons, method = 'nearest').to_dataframe()

    # assign ac capacity based on plant_number and index in plants_df
    df = df.merge(plants_df[['ac_capacity']], left_on='plant_number', right_index=True)

    # rename latitude, longitude columns to note that they came from the GEFS coordinates
    # rename time to add utc
    df = df.rename(columns={'latitude':'gefs_latitude','longitude':'gefs_longitude',
                            'time':'time_utc'
                })

    # remove index
    df = df.reset_index()

    # loop through unique forecasts (unique 'time_utc' and 'step' values)
    temp = {}
    i=0
    for time_utc in df['time_utc'].unique():
        for step in df['step'].unique():
            
            df_temp = df[(df['time_utc']==time_utc) & (df['step']==step)]
            out = df_temp.filter(regex='tcc').multiply(plants_df['ac_capacity'].values,axis='index').sum()/plants_df['ac_capacity'].sum()
            temp[i] = out.to_frame().T
            temp[i].insert(0, 'time_utc', time_utc)
            temp[i].insert(1, 'step', step)
            # because each time, step combination has a unique valid_time:
            valid_time = df_temp['valid_time'].unique()[0]
            # add valid_time
            temp[i].insert(2, 'valid_time', valid_time)
            i += 1

    # concatenate results
    weighted_avg_tcc = pd.concat(temp)
    del temp, df_temp, df

    # rename columns, set indices
    weighted_avg_tcc.rename(columns={'valid_time':'valid_time_utc_end_of_interval'},
                                    inplace=True)
    weighted_avg_tcc.set_index(['time_utc', 'step','valid_time_utc_end_of_interval'],inplace=True)

    # calculate stdev of weighted avg tcc for each valid_time_utc
    weighted_avg_tcc.insert(0,'tcc_std',weighted_avg_tcc.filter(regex='tcc').std(axis=1))

    weighted_avg_tcc = weighted_avg_tcc.reset_index().set_index('valid_time_utc_end_of_interval')

    # slice to region of interest
    # lat, lon extent for region of interest in GEFS coordinates
    # longitude has to be between 0 and 360
    min_lon = (region_extent[0]+360) % 360
    max_lon = (region_extent[1]+360) % 360
    min_lat = region_extent[2]
    max_lat = region_extent[3]

    # slice to region extent
    # note that lat is sliced max to min because of order that latitude is stored in with GEFS
    # (I think...)
    ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon,max_lon))
    # ds = ds.rename({'tcc':'tcc_' + member_list})
    # ds = ds.drop_vars(['number', 'atmosphere', 'gribfile_projection'])

    ds.close()
    return weighted_avg_tcc, ds

# def get_tcc_fcasts(init_time,fxx,plants_df,num_members=30,
#                    verbose=False,remove_grib=False,overwrite=False):
#     # list of GEFS ensemble members, e.g., 'p01', 'p02', etc.
#     member_list = [f"p{x:02d}" for x in range(1, num_members+1)]

#     # GEFS uses longitudes in [0,360), so use modulo to convert lon to be 0 <= lon < 360
#     lon_gefs = [(x + 360) % 360 for x in plants_df.longitude]

#     d={}
#     data=[]
#     member_site_tcc = np.empty([num_members, len(plants_df.index)])
#     for x in range(0, num_members):
#         d[x] = Herbie(init_time, model="gefs", product="atmos.5", 
#                             member=member_list[x], fxx=fxx,
#                             verbose=verbose).xarray("TCDC",
#                                                      remove_grib=remove_grib,
#                                                      overwrite=overwrite)
#         for y in range(0, len(plants_df.index)):
#             member_site_tcc[x][y] = d[x].sel(latitude=plants_df.latitude[y],
#                                                longitude=lon_gefs[y],
#                                                method='nearest').tcc.item()
#             # make a list
#             data.append((d[x].time.values,
#                         d[x].valid_time.values,
#                         y+1,
#                         plants_df.plant_name[y],
#                         plants_df.latitude[y],
#                         plants_df.longitude[y],
#                         d[x].sel(latitude=plants_df.latitude[y],longitude=lon_gefs[y],method='nearest').latitude.values.item(),
#                         d[x].sel(latitude=plants_df.latitude[y],longitude=lon_gefs[y],method='nearest').longitude.values.item(),
#                         x,
#                         member_site_tcc[x][y]))
#         # site-weighted average TCC for each ensemble member
#         weighted_avg_tcc = np.average(member_site_tcc[x], weights=plants_df.ac_capacity)
#         data.append((d[x].time.values,
#                         d[x].valid_time.values,
#                         'n/a', 
#                         'weighted_avg',
#                         'n/a',
#                         'n/a',
#                         'n/a',
#                         'n/a',
#                         x,
#                         weighted_avg_tcc))
#     #turn the list into a temporary dataframe
#     df_temp = pd.DataFrame(data, columns=('time_utc',
#                                         'valid_time_utc',
#                                         'site_number',
#                                         'site_name',
#                                         'site_lat',
#                                         'site_lon',
#                                         'model_lat',
#                                         'model_lon',
#                                         'member',
#                                         'tcc'
#                                         ))

#     # pivot the dataframe, https://stackoverflow.com/questions/35414625/pandas-how-to-run-a-pivot-with-a-multi-index
#     df = df_temp.pivot_table(
#         values='tcc',
#         index=['time_utc',
#                 'valid_time_utc',
#                 'site_number',
#                 'site_name',
#                 'site_lat',
#                 'site_lon',
#                 'model_lat',
#                 'model_lon'],
#         columns='member',
#         ).rename(columns=lambda x:f"p{x+1:02d}")

#     # calculate standard deviation of TCC across members
#     df['std'] = df.std(axis=1)
#     df['weight'] = plants_df.ac_capacity.to_list() + [np.nan] # add NaN for the weighted avg

#     # xarray dataset
#     ds = xr.concat([d[i] for i in d],dim='member')
#     # standard deviation across members
#     ds_std = ds.std(dim='member')

#     return df, ds, ds_std, data