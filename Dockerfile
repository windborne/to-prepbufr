FROM python:3.11

RUN mkdir "/app"

WORKDIR /app

# install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2.tar.gz &&  \
    tar -zxvf cmake-3.24.2.tar.gz &&  \
    cd cmake-3.24.2 && \
    ./bootstrap && \
    make &&  \
    make install

# install other dependencies for the NCEP bufr utils
RUN apt update &&  \
    apt install -y gfortran && \
    pip3 install numpy

# install NCEP bufr utils
RUN cd /app &&  \
    wget -c https://github.com/NOAA-EMC/NCEPLIBS-bufr/archive/refs/tags/bufr_v12.0.0.tar.gz -O - | tar -xz && \
    cd NCEPLIBS-bufr-bufr_v12.0.0 &&  \
    mkdir build &&  \
    cd build &&  \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/bin/ -DENABLE_PYTHON=ON .. && \
    make -j4 && \
    make install && \
    cd python && \
    pip3 install .

# install other python dependencies
RUN pip3 install pyjwt requests

COPY prepbufr_michael.table prepbufr_michael.table
COPY wb_to_prepbufr.py wb_to_prepbufr.py

CMD python3 wb_to_prepbufr.py