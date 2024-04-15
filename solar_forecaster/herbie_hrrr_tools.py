import pandas as pd
import pvlib
import xarray as xr
import numpy as np

from herbie import FastHerbie

import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import time
from pathlib import Path

def hrrr_ds_to_power(ds, plants_df, transposition_model = 'perez',
    decomposition_model = 'erbs', eta_inv_nom = 0.98):

    num_plants = len(plants_df) # number of plants
    
    # calculate temperature in celsius, wind speed
    ds = ds.assign(t2m_c=ds['t2m'] - 273.15)
    ds = ds.assign(wspd=np.sqrt(ds['u10']**2 + ds['v10']**2))

    # dataset setup
    df_hrrr_plants = ds.herbie.nearest_points(points=plants_df,
                                            names=plants_df['plant_code']) \
                                                .to_dataframe().reset_index()
    df_hrrr_plants = df_hrrr_plants.merge(plants_df[['plant_code','ac_capacity']],
                                        left_on='point', right_on='plant_code')
    df_hrrr_plants.drop(columns={'plant_code', 'surface', 'heightAboveGround', 
                                'metpy_crs', 'gribfile_projection'}, inplace=True)
    df_hrrr_plants.rename(columns={'time':'time_utc','valid_time':'valid_time_utc',
                                'point':'plant_code'},
                        inplace=True)

    # clear sky and solar position
    plant_code = plants_df['plant_code'][0]
    hrrr_data_temp = df_hrrr_plants.loc[df_hrrr_plants['plant_code'] == plant_code,
                                ['valid_time_utc','dswrf', 't2m_c','wspd']].copy()
    hrrr_data_temp = hrrr_data_temp.set_index('valid_time_utc')
    hrrr_data_temp.index = hrrr_data_temp.index.tz_localize(tz='UTC')
    clear_sky_dict_hrrr = {}
    solar_position_dict_hrrr = {}
    dni_extra = {}
    airmass = {}


    # power
    surface_tilt = {}
    surface_azimuth = {}
    total_irradiance = {}
    power_ac_hrrr = {}
    total_irradiance_cs = {}
    power_ac_hrrr_cs = {}
    for x in range(0,num_plants):
        plant_code = plants_df['plant_code'][x]
        # arrange weather data for plant
        hrrr_data = df_hrrr_plants.loc[df_hrrr_plants['plant_code'] == plant_code,
                                    ['valid_time_utc','dswrf', 't2m_c','wspd']].copy()
        hrrr_data = hrrr_data.set_index('valid_time_utc')
        hrrr_data.index = hrrr_data.index.tz_localize(tz='UTC')
        hrrr_data.rename(columns={'dswrf':'ghi', 't2m_c':'temp_air', 'wspd':'wind_speed'},
                        inplace=True)
        
        lat = plants_df.iloc[x]['latitude'] 
        lon = plants_df.iloc[x]['longitude']
        times = hrrr_data_temp.index 
        loc= pvlib.location.Location(latitude=lat, longitude=lon, tz=times.tz)
        solar_position_dict_hrrr[x] = loc.get_solarposition(times)
        # dni and airmass
        dni_extra[x] = pvlib.irradiance.get_extra_radiation(hrrr_data.index)
        airmass[x] = pvlib.atmosphere.get_relative_airmass(
            solar_position_dict_hrrr[x].apparent_zenith,
            # model='kastenyoung1989'
            model='gueymard2003'
        )

        # # simlified solis with aod700=0 and precipitable_water=0.5 appears to give good results
        # # goal is to set an upper-bound, not necessarily to be as accurate as possible
        clear_sky_dict_hrrr[x] = loc.get_clearsky(times,
                                                  model='simplified_solis',
                                                  dni_extra=dni_extra[x],
                                                  aod700 = 0.05,
                                                  precipitable_water = 0.5)
        # clear_sky_dict_hrrr[x] = loc.get_clearsky(times,
        #                                           model='ineichen',
        #                                           solar_position = solar_position_dict_hrrr[x],
        #                                           dni_extra=dni_extra[x],
        #                                         #   linke_turbidity=1
        #                                           )
        # clear_sky_dict_hrrr[x] = loc.get_clearsky(times)

        # decomposition
        if decomposition_model=='erbs':
            out_erbs = pvlib.irradiance.erbs(hrrr_data.ghi, solar_position_dict_hrrr[x].zenith,
                                                hrrr_data.index)
            hrrr_data['dni'] = out_erbs.dni
            hrrr_data['dhi'] = out_erbs.dhi
        elif decomposition_model=='dirint':
            dni_dirint = pvlib.irradiance.dirint(hrrr_data.ghi,
                                                    solar_position_dict_hrrr[x].zenith,
                                                    hrrr_data.index)
            df_dirint = pvlib.irradiance.complete_irradiance(
                solar_zenith=solar_position_dict_hrrr[x].apparent_zenith, ghi=hrrr_data.ghi,
                dni=dni_dirint, dhi=None
            )
            hrrr_data['dni'] = dni_dirint
            hrrr_data['dhi'] = df_dirint.dhi
        # surface tilt and azimuth
        if plants_df['tracking_type'][x] == 'single_axis':  
            # tracker orientation angles
            singleaxis_kwargs = dict(apparent_zenith=solar_position_dict_hrrr[x].apparent_zenith,
                                    apparent_azimuth=solar_position_dict_hrrr[x].azimuth,
                                    axis_tilt=plants_df['axis_tilt'][x],
                                    axis_azimuth=plants_df['axis_azimuth'][x],
                                    backtrack=plants_df['backtrack'][x],
                                    gcr=plants_df['ground_coverage_ratio'][x],
                                    )
            orientation = pvlib.tracking.singleaxis(max_angle=plants_df['max_rotation_angle'][x],
                                                    **singleaxis_kwargs)
            surface_tilt[x] = orientation.surface_tilt.fillna(0)
            surface_azimuth[x] = orientation.surface_azimuth.fillna(0)
        elif plants_df['tracking_type'][x] == 'fixed':
            surface_tilt[x] = float(plants_df['fixed_tilt'][x])
            surface_azimuth[x] = float(plants_df['fixed_azimuth'][x])

        # Transposed components of POA
        total_irradiance[x] = pvlib.irradiance.get_total_irradiance(
            surface_tilt[x], 
            surface_azimuth[x],
            solar_position_dict_hrrr[x].apparent_zenith, 
            solar_position_dict_hrrr[x].azimuth, 
            hrrr_data.dni, 
            hrrr_data.ghi, 
            hrrr_data.dhi, 
            airmass=airmass[x],
            albedo=0.2, # TODO: this could be a plant-specific value
            dni_extra=dni_extra[x], 
            model=transposition_model
        )

        # steady state cell temperature - faiman is much faster than fuentes, simpler than sapm
        t_cell = pvlib.temperature.faiman(total_irradiance[x].poa_global,
                                        hrrr_data.temp_air,
                                        hrrr_data.wind_speed,
        )

        # PVWatts dc power
        pdc = pvlib.pvsystem.pvwatts_dc(total_irradiance[x].poa_global,
                                        t_cell,
                                        plants_df['dc_capacity'][x],
                                        plants_df['temperature_coefficient'][x]/100)
        pdc0 = plants_df['ac_capacity'][x]/eta_inv_nom # inverter dc input is ac nameplate divided by nominal inverter efficiency
        pdc_inv = pdc*(1-plants_df['dc_loss_factor'][x]) # dc power into the inverter is modeled pdc after losses

        # PVWatts ac power
        power_ac_hrrr[x] = pvlib.inverter.pvwatts(pdc_inv, pdc0, eta_inv_nom)

        # clear sky
        # Transposed components of POA
        total_irradiance_cs[x] = pvlib.irradiance.get_total_irradiance(
            surface_tilt[x], 
            surface_azimuth[x],
            solar_position_dict_hrrr[x].apparent_zenith, 
            solar_position_dict_hrrr[x].azimuth, 
            clear_sky_dict_hrrr[x].dni, 
            clear_sky_dict_hrrr[x].ghi, 
            clear_sky_dict_hrrr[x].dhi, 
            airmass=airmass[x],
            albedo=0.2, 
            dni_extra=dni_extra[x], 
            model=transposition_model
        )

        # steady state cell temperature - faiman is much faster than fuentes, simpler than sapm
        t_cell = pvlib.temperature.faiman(total_irradiance_cs[x].poa_global,
                                        # temp_air = 5,
                                        # wind_speed = 1,
                                        hrrr_data.temp_air,
                                        hrrr_data.wind_speed,
        )

        # PVWatts dc power
        pdc = pvlib.pvsystem.pvwatts_dc(total_irradiance_cs[x].poa_global,
                                        t_cell,
                                        plants_df['dc_capacity'][x],
                                        plants_df['temperature_coefficient'][x]/100)
        pdc0 = plants_df['ac_capacity'][x]/eta_inv_nom # inverter dc input is ac nameplate divided by nominal inverter efficiency
        pdc_inv = pdc*(1-plants_df['dc_loss_factor'][x]) # dc power into the inverter is modeled pdc after losses

        # PVWatts ac power
        power_ac_hrrr_cs[x] = pvlib.inverter.pvwatts(pdc_inv, pdc0, eta_inv_nom)            

    power_ac_hrrr_all = pd.concat(power_ac_hrrr, axis=1, sort=False).sum(axis=1)
    power_ac_hrrr_all = power_ac_hrrr_all.to_frame(name='power_ac')
    power_ac_hrrr_all.index.names = ['time_center_labeled']

    power_ac_hrrr_all_cs = pd.concat(power_ac_hrrr_cs, axis=1, sort=False).sum(axis=1)
    power_ac_hrrr_all_cs = power_ac_hrrr_all_cs.to_frame(name='power_ac')
    power_ac_hrrr_all_cs.index.names = ['time_center_labeled']
    return power_ac_hrrr_all, power_ac_hrrr, power_ac_hrrr_all_cs, power_ac_hrrr_cs

def get_hrrr_fcasts_fastherbie(start, end, freq, fxx_range, plants_df,
                              region_extent=[0,1059,0,1799], save=True, 
                              save_dir='', attempts=5, transposition_model='perez',
                              decomposition_model='erbs', coarsen_type='none',
                              window_size=10):
    ymin = region_extent[0]
    ymax = region_extent[1]
    xmin = region_extent[2]
    xmax = region_extent[3]

    DATES = pd.date_range(start=start, end=end, freq=freq)

    filename = ('hrrr_solar_f' + str(min(fxx_range)) + '-f' + 
                    str(max(fxx_range)) + '_' + str(fxx_range.step) + 'h_' + 
                    DATES[0].strftime('%Y%m%d%H') + '-' + 
                    DATES[-1].strftime('%Y%m%d%H') + '_' + freq + '_step')
    the_file = Path(save_dir + filename + '.nc')
    if the_file.is_file():
        # try to load dataset from existing netcdf file
        # print('try loading ' + filename + '.nc')
        ds = xr.open_dataset(save_dir + filename + '.nc')
        print('loaded ' + filename + '.nc')
    else:
        ds_dict={}
        print('downloading new data')
        variables_list = ['DSWRF:surface','TMP:2 m','[U|V]GRD:10 m']
        FH = FastHerbie(DATES, model="HRRR", product="sfc", fxx=fxx_range)
        for i in range(0, len(variables_list)):
            for attempt in range(attempts):
                try:
                    if attempt==0:
                        # try downloading
                        ds_dict[i] = FH.xarray(variables_list[i],remove_grib=True)
                    else:
                        # after first attempt, set overwrite=True to overwrite partial files
                        ds_dict[i] = FH.xarray(variables_list[i],remove_grib=True,
                                            overwrite=True)
                except:
                    print('attempt ' + str(attempt+1) + ', pausing for ' + str((attempt+1)**2) + ' min')
                    time.sleep(60*(attempt+1)**2)
                else:
                    break
            else:
                raise ValueError('download failed, ran out of attempts')
                # return 'ERROR'
            print('variable group ' + str(i+1) + ' download complete')

        # merge datasets
        ds = xr.merge(ds_dict.values(), compat='override')
        # slice to region extent
        ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
        del ds_dict

        print('downloaded and processed ')
        if save==True:
            # save dataset as netcdf file
            # added encoding to fix issue with timestamps (?) being messed up
            # also reduces file size by about 50-80%
            comp = dict(zlib=True, complevel=1)
            encoding = {var: comp for var in ds.data_vars}
            # ds.to_netcdf(save_dir + filename + '.nc',
            #              encoding={'t2m': {'zlib': True, 'complevel': 1},
            #                        'dswrf': {'zlib': True, 'complevel': 1},
            #                        'u10': {'zlib': True, 'complevel': 1},
            #                        'v10': {'zlib': True, 'complevel': 1},})

            ds.to_netcdf(save_dir + filename + '.nc', encoding=encoding)
            print('saved ' + filename + '.nc')
        
    # coarsen, if needed:
    if coarsen_type=='mean':
        ds = ds.coarsen(x=window_size, y=window_size, boundary='trim').mean()
    elif coarsen_type=='max':
        ds = ds.coarsen(x=window_size, y=window_size, boundary='trim').max()
    elif coarsen_type=='min':
        ds = ds.coarsen(x=window_size, y=window_size, boundary='trim').min()

    p_ac_all, p_ac, p_ac_cs_all, p_ac_cs = hrrr_ds_to_power(ds,
                                    plants_df,
                                    transposition_model = transposition_model,
                                    decomposition_model = decomposition_model,
                                    eta_inv_nom = 0.98)
    return ds, p_ac_all, p_ac, p_ac_cs_all, p_ac_cs


def get_many_hrrr_fcasts_fastherbie(start, end, freq, query_len_days, fxx_range,
                                    plants_df, region_extent=[220,440,1080,1440],
                                    save=True, save_dir='', attempts=5,
                                    transposition_model='perez',
                                    decomposition_model='erbs',
                                    coarsen_type='none', window_size=10):
    end_dt = pd.to_datetime(end)
    current_start_dt = pd.to_datetime(start)
    # ds = {}
    # ds = xr.Dataset()
    p_ac_all = {}
    p_ac = {}
    p_ac_cs_all = {}
    p_ac_cs = {}
    i=0
    while current_start_dt <= end_dt:
        # define start and end time
        current_start = current_start_dt.strftime('%Y-%m-%dT%H:%M:%S')
        current_end_dt = current_start_dt + relativedelta(days=query_len_days) - pd.Timedelta(freq)
        current_end = current_end_dt.strftime('%Y-%m-%dT%H:%M:%S')
        # get data
        if i==0:
            ds, p_ac_all[i], _, p_ac_cs_all[i], _ = \
            get_hrrr_fcasts_fastherbie(current_start, current_end, freq,
                                       fxx_range,plants_df, region_extent,
                                       save,save_dir, attempts,
                                       transposition_model,decomposition_model,
                                       coarsen_type, window_size)
        else:
            ds_temp, p_ac_all[i], _, p_ac_cs_all[i], _ = \
                get_hrrr_fcasts_fastherbie(current_start, current_end, freq,
                                        fxx_range,plants_df, region_extent,
                                        save,save_dir, attempts,
                                        transposition_model,decomposition_model,
                                        coarsen_type, window_size)
            # ds = ds.combine_first(ds_temp)
            # ds = ds.merge(ds_temp)
            ds = xr.concat([ds, ds_temp], dim='time') # concat uses less memory than merge
        # increment start datetime and index
        current_start_dt = current_end_dt + pd.Timedelta(freq)
        i += 1


    # ds = xr.concat([ds[i] for i in ds], dim='time')
    p_ac_all = pd.concat(p_ac_all)
    p_ac_cs_all = pd.concat(p_ac_cs_all)
    # p_ac = pd.concat(p_ac)
    
    return ds, p_ac_all, p_ac_cs_all # , p_ac
