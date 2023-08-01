# to_prepbufr

This utility queries the [WindBorne API](https://windbornesystems.com/docs/api) and converts the files to prepbufr.
While it works out of the box, we encourage you to adapt it to your needs.

## Installing dependencies & , option 1: Docker
For your convenience, we have provided a Dockerfile that installs the requisite dependencies 

```bash
docker build -t to_prepbufr .
docker run --env WB_CLIENT_ID --env WB_API_KEY to_prepbufr
```

## Installing dependencies, option 2: from source

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

From here
