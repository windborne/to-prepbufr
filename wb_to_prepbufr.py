import os
import time
import datetime
import numpy as np
import jwt
import requests
import ncepbufr

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
In this section, we have the core functions to convert data to prepbufr
"""

TBASI = 273.15
TBASW = 273.15 + 100
ESBASW = 101324.6
ESBASI = 610.71


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


def convert_to_prepbufr(data, output_file='export.prepbufr'):
    if len(data) == 0:
        print("No data provided; skipping")
        return

    bufr = ncepbufr.open(output_file, 'w', table='prepbufr_config.table')

    hdstr = 'SID XOB YOB DHR TYP ELV SAID T29 TSB'
    obstr = 'POB QOB TOB ZOB UOB VOB PWO MXGS HOVI CAT PRSS TDO PMO'
    drstr = 'XDR YDR HRDR'
    qcstr = 'PQM QQM TQM ZQM WQM NUL PWQ PMQ'
    oestr = 'POE QOE TOE NUL WOE NUL PWE'

    start_date = datetime.datetime.utcfromtimestamp(data[0]['timestamp'])
    int_date = start_date.year * 1000000 + start_date.month * 10000 + start_date.day * 100 + start_date.hour
    subset = 'ADPUPA'

    hdr = bufr.missing_value * np.ones(len(hdstr.split()), float)

    for i in range(len(data)):
        point = data[i]
        delta_hours = (point['timestamp'] - data[0]['timestamp']) / 3600.0

        assert abs(delta_hours) <= 3 + 1e-5  # do not allow more than 3 hours of data
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
        obs[4, 0] = point['speed_x']
        obs[5, 0] = point['speed_y']
        qcf[4, 0] = 1.
        qcf[3, 0] = 1.
        qcf[1, 0] = 31.
        qcf[2, 0] = 31.

        # Right now, we simply hardcode these values to reasonable averages
        # However, in the future we may calculate these ourselves
        wind_x_error = 1.18
        wind_y_error = 1.18
        relative_humidity_error = 8.23
        temperature_error = 0.83

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
            point = {'altitude': 6791.21, 'humidity': 97.5297606332122, 'latitude': 55.37090461538461, 'longitude': 16.212832884615384, 'mission_name': 'W-696', 'pressure': 422.91500495374595, 'speed_x': 8.41, 'speed_y': 9.98, 'temperature': -24.819797888992383, 'timestamp': 1691082605}
            rdgas = 287.04
            rvgas = 461.50
            eps = rdgas / rvgas

            es = gfssvp(point['temperature']) * min(1, max(0, point['humidity'] / 100.))
            qs = eps * es / (point['pressure']*100.0 - (1 - eps) * es)
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


"""
In this section, we tie it all together, querying the WindBorne API and converting it to prepbufr
"""


def main():
    """
    Queries WindBorne API for data from the last three hours and converts it to prepbufr
    :return:
    """
    since = int(datetime.datetime.now().timestamp()) - 3 * 60 * 60
    observations_by_mission = {}

    while True:
        # Note that we query superobservations, which are described here:
        # https://windbornesystems.com/docs/api#super_observations
        # We find that for most NWP applications this leads to better performance than overwhelming with high-res data
        observations_page = wb_get_request(
            f"https://sensor-data.windbornesystems.com/api/v1/super_observations.json?since={since}&include_mission_name=true")

        print(f"Fetched page with {len(observations_page['observations'])} observation(s)")
        for observation in observations_page['observations']:
            if observation['mission_name'] not in observations_by_mission:
                observations_by_mission[observation['mission_name']] = []

            observations_by_mission[observation['mission_name']].append(observation)

        # if there's no new data, break
        if not observations_page['has_next_page']:
            break

            # alternatively, you could call `time.sleep(60)` and keep polling
            # (though you'd have to move where you were calling convert_to_prepbufr)

        # update since for the next request
        since = observations_page['next_since']

    if len(observations_by_mission) == 0:
        print("No observations found")
        return

    for mission_name, accumulated_observations in observations_by_mission.items():
        # make sure it's sorted
        accumulated_observations.sort(key=lambda x: x['timestamp'])

        # slice into 3 hour segments
        start_index = 0
        for i in range(len(accumulated_observations)):
            if accumulated_observations[i]['timestamp'] - accumulated_observations[start_index]['timestamp'] > 3 * 60 * 60:
                segment = accumulated_observations[start_index:i]
                output_file = f"windborne_data_{segment[0]['timestamp']}.prepbufr"

                print(f"Converting {len(segment)} observation(s) to prepbufr and saving as {output_file}")
                convert_to_prepbufr(segment, output_file)

                start_index = i

        segment = accumulated_observations[start_index:]
        output_file = f"windborne_data_{mission_name}_{segment[0]['timestamp']}.prepbufr"
        print(f"Converting {len(segment)} observation(s) to prepbufr and saving as {output_file}")
        convert_to_prepbufr(segment, output_file)


if __name__ == '__main__':
    main()
