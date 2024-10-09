import os
import time
import datetime
import numpy as np
import jwt
import requests
import ncepbufr
import argparse
import xarray as xr
import pandas as pd

"""
In this section, we define the helper functions to access the WindBorne API
This is described in https://windbornesystems.com/docs/api
"""


def wb_get_request(url):
    """
    Make a GET request to WindBorne, authorizing with WindBorne correctly
    """

    client_id = os.environ['WB_CLIENT_ID']  # Make sure to set this!
    api_key = os.environ['WB_API_KEY']  # Make sure to set this!

    # create a signed JSON Web Token for authentication
    # this token is safe to pass to other processes or servers if desired, as it does not expose the API key
    signed_token = jwt.encode({
        'client_id': client_id,
        'iat': int(time.time()),
    }, api_key, algorithm='HS256')

    # make the request, checking the status code to make sure it succeeded
    response = requests.get(url, auth=(client_id, signed_token))
    response.raise_for_status()

    # return the response body
    return response.json()

"""
These are the observation error values for radiosondes that were taken 
from NCEP'S GSI data tables.  

The first table has error values for the thermodynamics (P, T, RH)
column 1: pressure level (hPa),
column 2: temperature error (K),
column 3: RH error (tens of %, so, 0.20E+01 is 20%)
other columns: irrelevant
"""
error_thermo_str = """
  0.11000E+04 0.12000E+01 0.20000E+01 0.10000E+10 0.11000E+01 0.10000E+10
  0.10500E+04 0.12000E+01 0.20000E+01 0.10000E+10 0.11000E+01 0.10000E+10
  0.10000E+04 0.12000E+01 0.20000E+01 0.10000E+10 0.11000E+01 0.10000E+10
  0.95000E+03 0.11000E+01 0.20000E+01 0.10000E+10 0.11000E+01 0.10000E+10
  0.90000E+03 0.90000E+00 0.20000E+01 0.10000E+10 0.11000E+01 0.10000E+10
  0.85000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.11000E+01 0.10000E+10
  0.80000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.11000E+01 0.10000E+10
  0.75000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.12000E+01 0.10000E+10
  0.70000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.12000E+01 0.10000E+10
  0.65000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.12000E+01 0.10000E+10
  0.60000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.12000E+01 0.10000E+10
  0.55000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.50000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.45000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.40000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.35000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.30000E+03 0.90000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.25000E+03 0.12000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.20000E+03 0.12000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.15000E+03 0.10000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.10000E+03 0.80000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.75000E+02 0.80000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.50000E+02 0.90000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.40000E+02 0.95000E+00 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.30000E+02 0.10000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.20000E+02 0.12500E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.10000E+02 0.15000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.50000E+01 0.15000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.40000E+01 0.15000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.30000E+01 0.15000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.20000E+01 0.15000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.10000E+01 0.15000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10
  0.00000E+00 0.15000E+01 0.20000E+01 0.10000E+10 0.10000E+10 0.10000E+10"""

"""
These are the errors for winds:
column 1: Pressure (hPa)
column 4: Wind Error (m/s)
columns 2-3, 5-6: Irrelevant
"""
error_winds_str = """
  0.11000E+04 0.10000E+10 0.10000E+10 0.14000E+01 0.10000E+10 0.10000E+10
  0.10500E+04 0.10000E+10 0.10000E+10 0.14000E+01 0.10000E+10 0.10000E+10
  0.10000E+04 0.10000E+10 0.10000E+10 0.14000E+01 0.10000E+10 0.10000E+10
  0.95000E+03 0.10000E+10 0.10000E+10 0.15000E+01 0.10000E+10 0.10000E+10
  0.90000E+03 0.10000E+10 0.10000E+10 0.15000E+01 0.10000E+10 0.10000E+10
  0.85000E+03 0.10000E+10 0.10000E+10 0.15000E+01 0.10000E+10 0.10000E+10
  0.80000E+03 0.10000E+10 0.10000E+10 0.16000E+01 0.10000E+10 0.10000E+10
  0.75000E+03 0.10000E+10 0.10000E+10 0.16000E+01 0.10000E+10 0.10000E+10
  0.70000E+03 0.10000E+10 0.10000E+10 0.16000E+01 0.10000E+10 0.10000E+10
  0.65000E+03 0.10000E+10 0.10000E+10 0.18000E+01 0.10000E+10 0.10000E+10
  0.60000E+03 0.10000E+10 0.10000E+10 0.19000E+01 0.10000E+10 0.10000E+10
  0.55000E+03 0.10000E+10 0.10000E+10 0.20000E+01 0.10000E+10 0.10000E+10
  0.50000E+03 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.45000E+03 0.10000E+10 0.10000E+10 0.23000E+01 0.10000E+10 0.10000E+10
  0.40000E+03 0.10000E+10 0.10000E+10 0.26000E+01 0.10000E+10 0.10000E+10
  0.35000E+03 0.10000E+10 0.10000E+10 0.28000E+01 0.10000E+10 0.10000E+10
  0.30000E+03 0.10000E+10 0.10000E+10 0.30000E+01 0.10000E+10 0.10000E+10
  0.25000E+03 0.10000E+10 0.10000E+10 0.32000E+01 0.10000E+10 0.10000E+10
  0.20000E+03 0.10000E+10 0.10000E+10 0.27000E+01 0.10000E+10 0.10000E+10
  0.15000E+03 0.10000E+10 0.10000E+10 0.24000E+01 0.10000E+10 0.10000E+10
  0.10000E+03 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.75000E+02 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.50000E+02 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.40000E+02 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.30000E+02 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.20000E+02 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.10000E+02 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.50000E+01 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.40000E+01 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.30000E+01 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.20000E+01 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.10000E+01 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10
  0.00000E+00 0.10000E+10 0.10000E+10 0.21000E+01 0.10000E+10 0.10000E+10"""

"""
Some quick parsing here, in order to get the data into variables
"""
error_thermo_vals = np.fromstring(error_thermo_str, sep=' ')
error_thermo_vals.shape = (len(error_thermo_vals)//6, 6)
error_thermo_vals = np.flipud(error_thermo_vals)
error_thermo = {}
error_thermo["pressure"] = error_thermo_vals[:, 0]
error_thermo["temp"] = error_thermo_vals[:, 1]
error_thermo["rh"] = error_thermo_vals[:, 2] * 10

error_winds_vals = np.fromstring(error_winds_str, sep=' ')
error_winds_vals.shape = (len(error_winds_vals)//6, 6)
error_winds_vals = np.flipud(error_winds_vals)
error_winds = {}
error_winds["pressure"] = error_winds_vals[:, 0]
error_winds["speed"] = error_winds_vals[:, 3]
assert (error_thermo["pressure"] == error_winds["pressure"]).all()

"""
In this section, we have the core functions to convert data to prepbufr
"""

TBASI = 273.15
TBASW = 273.15 + 100
ESBASW = 101324.6
ESBASI = 610.71

RDGAS = 287.04
RVGAS = 461.50
EPS = RDGAS / RVGAS


@np.vectorize
def gfssvp(temperature_celsius):
    """
    Calculate saturation vapor pressure with GFS formula
    Note this is slightly different from other formulas you'll see
    So, you may wish to replace this function, or to replace to relative humidity -> specific humidity calc entirely
    """

    temperature_kelvin = temperature_celsius + 273.15
    esice = 0

    if temperature_kelvin < TBASI:
        x = -9.09718 * (TBASI / temperature_kelvin - 1.0) - 3.56654 * np.log10(
            TBASI / temperature_kelvin) + 0.876793 * (1.0 - temperature_kelvin / TBASI) + np.log10(
            ESBASI)
        esice = 10. ** x

    esh2o = 0
    if temperature_kelvin > TBASI - 20:
        x = -7.90298 * (TBASW / temperature_kelvin - 1.0) + 5.02808 * np.log10(TBASW / temperature_kelvin) \
            - 1.3816e-07 * (10.0 ** ((1.0 - temperature_kelvin / TBASW) * 11.344) - 1.0) \
            + 8.1328e-03 * (10.0 ** ((TBASW / temperature_kelvin - 1.0) * (-3.49149)) - 1.0) \
            + np.log10(ESBASW)
        esh2o = 10. ** x

    if temperature_kelvin <= -20 + TBASI:
        es = esice
    elif temperature_kelvin >= TBASI:
        es = esh2o
    else:
        es = 0.05 * ((TBASI - temperature_kelvin) * esice + (temperature_kelvin - TBASI + 20.) * esh2o)
    return es


def convert_to_prepbufr(data, reftime, output_file='export.prepbufr'):
    if len(data) == 0:
        print("No data; skipping")
        return

    bufr = ncepbufr.open(output_file, 'w', table='prepbufr_config.table')

    hdstr = 'SID XOB YOB DHR TYP ELV SAID T29 TSB'
    obstr = 'POB QOB TOB ZOB UOB VOB PWO MXGS HOVI CAT PRSS TDO PMO'
    drstr = 'XDR YDR HRDR'
    qcstr = 'PQM QQM TQM ZQM WQM NUL PWQ PMQ'
    oestr = 'POE QOE TOE NUL WOE NUL PWE'

    start_date = datetime.datetime.fromtimestamp(reftime, tz=datetime.timezone.utc)
    int_date = start_date.year * 1000000 + start_date.month * 10000 + start_date.day * 100 + start_date.hour
    subset = 'ADPUPA'

    hdr = bufr.missing_value * np.ones(len(hdstr.split()), float)

    for i in range(len(data)):
        point = data[i]
        delta_hours = (point['timestamp'] - data[0]['timestamp']) / 3600.0
        delta_hours = (point['timestamp'] - reftime) / 3600.0

        assert i == 0 or data[i - 1]['timestamp'] <= point['timestamp']  # do not allow out of order data

        hdr[:] = bufr.missing_value

        hdr[0] = np.frombuffer(point['mission_name'].ljust(8)[:8].encode(), dtype=np.float64)[0]
        hdr[1] = point['longitude']
        hdr[2] = point['latitude']
        hdr[3] = delta_hours
        hdr[4] = 232
        hdr[8] = 1

        bufr.open_message(subset, int_date)

        nlvl = 1

        obs = bufr.missing_value * np.ones((len(obstr.split()), nlvl), float)
        dr = bufr.missing_value * np.ones((len(drstr.split()), nlvl), float)
        dr[0] = hdr[1]
        dr[1] = hdr[2]
        dr[2] = hdr[3]
        oer = bufr.missing_value * np.ones((len(oestr.split()), nlvl), float)
        qcf = bufr.missing_value * np.ones((len(qcstr.split()), nlvl), float)

        if point['pressure'] is None:
            qcf[0, 0] = 31.
        else:
            obs[0, 0] = point['pressure']
            qcf[0, 0] = 1.

        obs[3, 0] = point['altitude']
        obs[4, 0] = point['speed_u']
        obs[5, 0] = point['speed_v']
        qcf[4, 0] = 1.
        qcf[3, 0] = 1.
        qcf[1, 0] = 31.
        qcf[2, 0] = 31.

        # Set the error values using the input table data that is above.
        if point["pressure"] == None:
            # Just a quick estimate, close enough for error characteristics
            if (point["altitude"] == None):
                print("Warning: Found some data with no pressure or altitude, skipping.")
            interp_pressure = 3.83325e-22 * (44330.7 - point["altitude"])**5.255799
        else:
            interp_pressure = point["pressure"]
        wind_speed_error = np.interp(interp_pressure, error_winds["pressure"], error_winds["speed"])
        wind_x_error = wind_speed_error
        wind_y_error = wind_speed_error
        relative_humidity_error = np.interp(interp_pressure, error_thermo["pressure"], error_thermo["rh"])
        temperature_error = np.interp(interp_pressure, error_thermo["pressure"], error_thermo["temp"])

        oer[3, 0] = 4
        oer[4, 0] = max(wind_x_error, wind_y_error)

        bufr.write_subset(hdr, hdstr)
        bufr.write_subset(obs, obstr)
        bufr.write_subset(dr, drstr)
        bufr.write_subset(oer, oestr)
        bufr.write_subset(qcf, qcstr, end=True)

        # begin second subblock

        hdr[4] = 132
        hdr[8] = 1
        obs[4:, 0] = bufr.missing_value
        qcf[4:, 0] = bufr.missing_value
        qcf[4, 0] = 31.
        oer[:, 0] = bufr.missing_value
        oer[3, 0] = 4

        # convert from relative humidity to specific humidity
        specific_humidity = None
        if point['temperature'] is not None and point['pressure'] is not None and point['humidity'] is not None:
            #point = {'altitude': 6791.21, 'humidity': 97.5297606332122, 'latitude': 55.37090461538461, 'longitude': 16.212832884615384, 'mission_name': 'W-696', 'pressure': 422.91500495374595, 'speed_x': 8.41, 'speed_y': 9.98, 'temperature': -24.819797888992383, 'timestamp': 1691082605}

            es = gfssvp(point['temperature']) * min(1, max(0, point['humidity'] / 100.))
            qs = EPS * es / (point['pressure'] * 100.0 - (1 - EPS) * es)
            newspec = qs

            spec = newspec

            specific_humidity = spec * 1e6  # in mg/kg

        if specific_humidity is not None:
            obs[1, 0] = specific_humidity
            oer[1, 0] = relative_humidity_error * 0.7
            qcf[1, 0] = 2.
        else:
            qcf[1, 0] = 31.

        if point['temperature'] is not None:
            obs[2, 0] = point['temperature']
            oer[2, 0] = temperature_error
            qcf[2, 0] = 1.
        else:
            qcf[2, 0] = 31.

        bufr.write_subset(hdr, hdstr)
        bufr.write_subset(obs, obstr)
        bufr.write_subset(dr, drstr)
        bufr.write_subset(oer, oestr)
        bufr.write_subset(qcf, qcstr, end=True)

        bufr.close_message()

    bufr.close()

def convert_to_netcdf(data, curtime, bucket_hours ):
    # This module outputs data in netcdf format for the WMO ISARRA program.  The output format is netcdf
    #   and the style (variable names, file names, etc.) are described here:
    #  https://github.com/synoptic/wmo-uasdc/tree/main/raw_uas_to_netCDF

    # Mapping of WindBorne names to ISARRA names
    rename_dict = {
        'latitude' : 'lat',
        'longitude' : 'lon',
        'altitude' : 'altitude',
        'temperature' : 'air_temperature',
        'wind_direction' : 'wind_direction',
        'wind_speed' : 'wind_speed',
        'pressure' : 'air_pressure',
        'humidity_mixing_ratio' : 'humidity_mixing_ratio',
        'index' : 'obs',
    }

    # Put the data in a panda datafram in order to easily push to xarray then netcdf output
    df = pd.DataFrame(data)
    ds = xr.Dataset.from_dataframe(df)

    # Build the filename and save some variables for use later
    mt = datetime.datetime.fromtimestamp(curtime, tz=datetime.timezone.utc)
    outdatestring = mt.strftime('%Y%m%d%H%M%S')
    mission_name = ds['mission_name'].data[0]
    output_file = 'USADC_300_0{}_{}Z.nc'.format(mission_name[2:6],outdatestring)

    # Derived quantities calculated here:

    # convert from specific humidity to humidity_mixing_ratio
    mg_to_kg = 1000000.
    if not all(x is None for x in ds['specific_humidity'].data):
        ds['humidity_mixing_ratio'] = (ds['specific_humidity'] / mg_to_kg) / (1 - (ds['specific_humidity'] / mg_to_kg))
    else:
        ds['humidity_mixing_ratio'] = ds['specific_humidity']

    # Wind speed and direction from components
    ds['wind_speed'] = np.sqrt(ds['speed_u']*ds['speed_u'] + ds['speed_v']*ds['speed_v'])
    ds['wind_direction'] = np.mod(180 + (180 / np.pi) * np.arctan2(ds['speed_u'], ds['speed_v']), 360)

    ds['time'] = ds['timestamp'].astype(float)
    ds = ds.assign_coords(time=("time", ds['time'].data))

    # Now that calculations are done, remove variables not needed in the netcdf output
    ds = ds.drop_vars(['humidity', 'speed_u', 'speed_v', 'speed_x', 'speed_y', 'specific_humidity',
                       'timestamp', 'mission_name'])

    # Rename the variables
    ds = ds.rename(rename_dict)

    # Adding attributes to variables in the xarray dataset
    ds['time'].attrs = {'units': 'seconds since 1970-01-01T00:00:00', 'long_name': 'Time', '_FillValue': float('nan'),
                        'processing_level': ''}
    ds['lat'].attrs = {'units': 'degrees_north', 'long_name': 'Latitude', '_FillValue': float('nan'),
                       'processing_level': ''}
    ds['lon'].attrs = {'units': 'degrees_east', 'long_name': 'Longitude', '_FillValue': float('nan'),
                       'processing_level': ''}
    ds['altitude'].attrs = {'units': 'meters_above_sea_level', 'long_name': 'Altitude', '_FillValue': float('nan'),
                            'processing_level': ''}
    ds['air_temperature'].attrs = {'units': 'Kelvin', 'long_name': 'Air Temperature', '_FillValue': float('nan'),
                                   'processing_level': ''}
    ds['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind Speed', '_FillValue': float('nan'),
                              'processing_level': ''}
    ds['wind_direction'].attrs = {'units': 'degrees', 'long_name': 'Wind Direction', '_FillValue': float('nan'),
                                  'processing_level': ''}
    ds['humidity_mixing_ratio'].attrs = {'units': 'kg/kg', 'long_name': 'Humidity Mixing Ratio',
                                         '_FillValue': float('nan'), 'processing_level': ''}
    ds['air_pressure'].attrs = {'units': 'Pa', 'long_name': 'Atmospheric Pressure', '_FillValue': float('nan'),
                                'processing_level': ''}

    # Add Global Attributes synonymous across all UASDC providers
    ds.attrs['Conventions'] = "CF-1.8, WMO-CF-1.0"
    ds.attrs['wmo__cf_profile'] = "FM 303-2024"
    ds.attrs['featureType'] = "trajectory"

    # Add Global Attributes unique to Provider
    ds.attrs['platform_name'] = "WindBorne Global Sounding Balloon"
    ds.attrs['flight_id'] = mission_name
    ds.attrs['site_terrain_elevation_height'] = 'not applicable'
    ds.attrs['processing_level'] = "b1"
    ds.to_netcdf(output_file)

def output_data(accumulated_observations, mission_name, starttime, bucket_hours, netcdf_output=False):
    accumulated_observations.sort(key=lambda x: x['timestamp'])

    # Here, set the earliest time of data to be the first observation time, then set it to the most recent
    #    start of a bucket increment.
    # The reason to do this rather than using the input starttime, is because sometimes the data
    #    doesn't start at the start time, and the underlying output would try to output data that doesn't exist
    #
    accumulated_observations.sort(key=lambda x: x['timestamp'])
    earliest_time = accumulated_observations[0]['timestamp']
    if (earliest_time < starttime):
        print("WTF, how can we have gotten data from before the starttime?")
    curtime = earliest_time - earliest_time % (bucket_hours * 60 * 60)

    start_index = 0
    for i in range(len(accumulated_observations)):
        if accumulated_observations[i]['timestamp'] - curtime > bucket_hours * 60 * 60:
            segment = accumulated_observations[start_index:i]
            mt = datetime.datetime.fromtimestamp(curtime, tz=datetime.timezone.utc)+datetime.timedelta(hours=bucket_hours/2)
            output_file = (f"WindBorne_%s_%04d-%02d-%02d_%02d:00_%dh.prepbufr" %
                           (mission_name, mt.year, mt.month, mt.day, mt.hour, bucket_hours))
            if (netcdf_output):
                print(f"Converting {len(segment)} observation(s) and saving as netcdf")
                convert_to_netcdf(segment, curtime, bucket_hours)
            else:
                print(f"Converting {len(segment)} observation(s) to prepbufr and saving as {output_file}")
                convert_to_prepbufr(segment, curtime + datetime.timedelta(hours=bucket_hours/2).seconds, output_file)

            start_index = i
            curtime += datetime.timedelta(hours=bucket_hours).seconds

    # Cover any extra data within the latest partial bucket
    segment = accumulated_observations[start_index:]
    mt = datetime.datetime.fromtimestamp(curtime, tz=datetime.timezone.utc) + datetime.timedelta(hours=bucket_hours / 2)
    output_file = (f"WindBorne_%s_%04d-%02d-%02d_%02d:00_%dh.prepbufr" %
                   (mission_name, mt.year, mt.month, mt.day, mt.hour, bucket_hours))
    if (netcdf_output):
        print(f"Converting {len(segment)} observation(s) and saving as netcdf")
        convert_to_netcdf(segment, curtime, bucket_hours)
    else:
        print(f"Converting {len(segment)} observation(s) to prepbufr and saving as {output_file}")
        convert_to_prepbufr(segment, curtime + datetime.timedelta(hours=bucket_hours / 2).seconds, output_file)

def main():
    """
    Queries WindBorne API for data from the input time range and converts it to prepbufr
    :return:
    """

    parser = argparse.ArgumentParser(description="""
    Retrieves WindBorne data and output to prep bufr format.
    Outputs files that meet NCEP's prepbufr format.
    
    Files will be broken up into time buckets as specified by the --bucket_hours option, 
    and the output file names will contain the time at the mid-point of the bucket. For 
    example, if you are looking to have files centered on say, 00 UTC 29 April, the start time
    should be 3 hours prior to 00 UTC, 21 UTC 28 April.
    """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("times", nargs='+',
                        help='Starting and ending times to retrieve obs.  Format: YY-mm-dd_HH:MM '
                             'Ending time is optional, with current time used as default')
    parser.add_argument('-b', '--bucket_hours', type=float, default=6.0,
                        help='Number of hours of observations to accumulate into a file before opening the next file')
    parser.add_argument('-c', '--combine_missions', action='store_true',
                        help="If selected, all missions are combined in the same output file, only used for bufr.")
    parser.add_argument('-nc', '--netcdf_output', action='store_true',
                        help="If selected, data is output in netcdf format following conventions for ISARRA.")
    args = parser.parse_args()

    if (len(args.times) == 1):
        starttime=int(datetime.datetime.strptime(args.times[0], '%Y-%m-%d_%H:%M').
                   replace(tzinfo=datetime.timezone.utc).timestamp())
        endtime=int(datetime.datetime.now().timestamp())
    elif (len(args.times) == 2):
        starttime=int(datetime.datetime.strptime(args.times[0], '%Y-%m-%d_%H:%M').
                   replace(tzinfo=datetime.timezone.utc).timestamp())
        endtime=int(datetime.datetime.strptime(args.times[1], '%Y-%m-%d_%H:%M').
                 replace(tzinfo=datetime.timezone.utc).timestamp())
    else:
        print("error processing input args, one or two arguments are needed")
        exit(1)

    if (not "WB_CLIENT_ID" in os.environ) or (not "WB_API_KEY" in os.environ) :
        print("  ERROR: You must set environment variables WB_CLIENT_ID and WB_API_KEY\n"
              "  If you don't have a client ID or API key, please contact WindBorne.")
        exit(1)

    args = parser.parse_args()
    bucket_hours = args.bucket_hours

    observations_by_mission = {}
    accumulated_observations = []
    has_next_page = True

    # This line here would just find W-1594, useful for testing/debugging
    #next_page = f"https://sensor-data.windbornesystems.com/api/v1/super_observations.json?mission_id=c8108dd5-bcf5-45ec-be80-a1da5e382e99&min_time={starttime}&max_time={endtime}&include_mission_name=true"

    next_page = f"https://sensor-data.windbornesystems.com/api/v1/super_observations.json?min_time={starttime}&max_time={endtime}&include_mission_name=true"
    netcdf_output = args.netcdf_output

    while has_next_page:
        # Note that we query superobservations, which are described here:
        # https://windbornesystems.com/docs/api#super_observations
        # We find that for most NWP applications this leads to better performance than overwhelming with high-res data
        print(next_page)
        observations_page = wb_get_request(next_page)
        has_next_page = observations_page["has_next_page"]
        if (len(observations_page['observations']) == 0):
            print("Could not find any observations for the input date range!!!!")
        if has_next_page:
            next_page = observations_page["next_page"]+"&include_mission_name=true&min_time={}&max_time={}".format(starttime,endtime)
        print(f"Fetched page with {len(observations_page['observations'])} observation(s)")
        for observation in observations_page['observations']:
            if 'mission_name' not in observation:
                print("got an ob without a mission name???")
                continue
            elif observation['mission_name'] not in observations_by_mission:
                observations_by_mission[observation['mission_name']] = []

            observations_by_mission[observation['mission_name']].append(observation)
            accumulated_observations.append(observation)

            # alternatively, you could call `time.sleep(60)` and keep polling here
            # (though you'd have to move where you were calling convert_to_prepbufr)


    if len(observations_by_mission) == 0:
        print("No observations found")
        return

    if (args.combine_missions):
        mission_name = 'all'
        output_data(accumulated_observations, mission_name, starttime, bucket_hours)
    else:
        for mission_name, accumulated_observations in observations_by_mission.items():
           output_data(accumulated_observations, mission_name, starttime, bucket_hours, netcdf_output)

if __name__ == '__main__':
    main()
