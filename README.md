# to_prepbufr

This utility queries the [WindBorne API](https://windbornesystems.com/docs/api) and converts the files to prepbufr.
While it works out of the box, we encourage you to adapt it to your needs.

For both methods of running, you will need to set the environment variables `WB_CLIENT_ID` and `WB_API_KEY`.
If you do not have these, you may request them by emailing data@windbornesystems.com.

## Installing dependencies & running, option 1: Docker
For your convenience, we have provided a Dockerfile that installs the requisite dependencies.
To build and run it:

```bash
docker build -t to_prepbufr .
docker run --env WB_CLIENT_ID --env WB_API_KEY -it to_prepbufr bash
python3 wb_to_prepbufr.py
```

Note that, as it's in a docker container, it is isolated and you will not be able to access the output files from your normal shell.

## Installing dependencies & running, option 2: from source

As well as the NCEPLIBS-bufr lib described below, you will need the following other dependencies (which may well exist on your system):
1. cmake (https://cmake.org/download/)
2. python3 (https://www.python.org/downloads/)
3. numpy (`pip3 install numpy`)
4. gfortran (`apt install gfortran`)

With these in hand, you will need the NCEPLIBS-bufr lib. 
Follow the instructions at [https://github.com/NOAA-EMC/NCEPLIBS-bufr](https://github.com/NOAA-EMC/NCEPLIBS-bufr), making sure to pass -DENABLE_PYTHON=ON. 
For the sake of convenience, they are summarized here as:
1. Download and untar https://github.com/NOAA-EMC/NCEPLIBS-bufr/archive/refs/tags/bufr_v12.0.0.tar.gz
2. Enter that folder and `mkdir build && cd build`
3. Prepare the makefile with python integrations enabled and installing to `/usr/local/bin` with `cmake -DCMAKE_INSTALL_PREFIX=/usr/local/bin/ -DENABLE_PYTHON=ON ..`
4. `make -j4`
5. `make install`
6. Install the generated python package with `cd python && pip3 install .` (note that this is not described in the NCEPLIBS-bufr documentation, but is necessary nonetheless)  

From here, you should be able to go back to wherever this repository lives and run:
```bash
python3 wb_to_prepbufr.py
```

## Assumptions
This utility is designed to be adapted to specific applications.
In the course of building it, we made several assumptions which may not be suited for your particular application, including:
- The formula for converting relative humidity to specific humidity. It uses formulas from GFS, which differ from formulas you may see elsewhere (eg metpy).
- How it divides up data to put in different files. It splits by balloon and by time period, such that a single file won't have more than three hours of data nor data from different balloon flights. It may make sense in some cases to reduce this time period.
- How much data it fetches from the WindBorne API. It is currently set to process only the last three hours of data and not to continue polling for more.
- The bufr codes: it currently uses 132 for temperature and humidity, and 232 for pressure and winds, corresponding to ADPUPA. Alternative codes made be better suited depending on the data assimilation setup.
