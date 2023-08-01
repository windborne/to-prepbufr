import ncepbufr

BAD = -999999


@np.vectorize
def gfssvp(t_):
    t = t_+273.15
    esice = 0
    TBASI = 273.15
    TBASW = 273.15 + 100
    ESBASW = 101324.6
    ESBASI = 610.71
    if t < TBASI:
         x = -9.09718*(TBASI/t-1.0) - 3.56654*np.log10(TBASI/t) +0.876793*(1.0-t/TBASI) + np.log10(ESBASI)
         esice =10.**(x)

    esh2o = 0
    if t > TBASI - 20:
        x = -7.90298*(TBASW/t-1.0) + 5.02808*np.log10(TBASW/t) \
             -1.3816e-07*(10.0**((1.0-t/TBASW)*11.344)-1.0)        \
             +8.1328e-03*(10.0**((TBASW/t-1.0)*(-3.49149))-1.0)    \
             +np.log10(ESBASW)
        esh2o = 10.**(x)

    if t <= -20 + TBASI:
        es = esice
    elif t >= TBASI:
        es = esh2o
    else:
        es = 0.05*((TBASI-t)*esice + (t-TBASI+20.)*esh2o)
    return es


def export_bufr2(out, data, reft0):
    if len(data["unix"]) == 0:
        print("no data for", out, "passing")
        return

    hdstr='SID XOB YOB DHR TYP ELV SAID T29 TSB'
    obstr='POB QOB TOB ZOB UOB VOB PWO MXGS HOVI CAT PRSS TDO PMO'
    drstr='XDR YDR HRDR'
    qcstr='PQM QQM TQM ZQM WQM NUL PWQ PMQ'
    oestr='POE QOE TOE NUL WOE NUL PWE'

    bufr = ncepbufr.open('mst4/bufr_wbprep/'+out+'.prepbufr','w',table='/home/joan/windborne/meteo/neoreal/prepbufr_michael.table')

    date = datetime(1970,1,1)+timedelta(seconds=reft0)
    idate = date.year * 1000000 + date.month * 10000 + date.day * 100 + date.hour
    subset='ADPUPA'
    hdr = bufr.missing_value*np.ones(len(hdstr.split()),np.float)

    biascycl = []
    L = len(data["unix"])
    for i in range(L):
        dh = (data["unix"][i] - reft0)/3600
        assert abs(dh) <= 3+1e-5
        fn = int(str(data["flight_name@MetaData"][i]).split("-")[1])
        hdr[:]=bufr.missing_value
        hdr[0] = np.fromstring('W-%3d   ' % fn,dtype=np.float64)[0]
        lonx = data["longitude@MetaData"][i]
        hdr[1]=lonx; hdr[2]=data["latitude@MetaData"][i]; hdr[3]=dh

        bufr.open_message(subset, idate)

        nlvl=1
        hdr[4]=220
        hdr[4] = 232
        #hdr[7] = 31
        hdr[8] = 1
        obs = bufr.missing_value*np.ones((len(obstr.split()),nlvl),np.float)
        dr = bufr.missing_value*np.ones((len(drstr.split()),nlvl),np.float)
        dr[0] = hdr[1]
        dr[1] = hdr[2]
        dr[2] = hdr[3]
        oer = bufr.missing_value*np.ones((len(oestr.split()),nlvl),np.float)
        qcf = bufr.missing_value*np.ones((len(qcstr.split()),nlvl),np.float)
        pr = data["air_pressure@MetaData"][i]
        if pr == BAD:
            qcf[0,0] = 31.
        else:
            obs[0,0]=pr * 0.01
            qcf[0,0] = 1.
        obs[3,0]=data["height@MetaData"][i]
        obs[4,0]=data["eastward_wind@ObsValue"][i]
        obs[5,0]=data["northward_wind@ObsValue"][i]
        qcf[4,0] = 1.
        qcf[3,0] = 1.
        qcf[1,0] = 31.
        qcf[2,0] = 31.

        ue = data["eastward_wind@ObsError"][i]
        ve = data["northward_wind@ObsError"][i]
        
        oer[3,0] = 4
        oer[4,0] = max(ue, ve)

        bufr.write_subset(hdr,hdstr)
        bufr.write_subset(obs,obstr)
        bufr.write_subset(dr,drstr)
        bufr.write_subset(oer,oestr)
        bufr.write_subset(qcf,qcstr,end=True)

        hdr[4] = 120
        hdr[4] = 132
        hdr[8] = 1
        obs[4:,0] = bufr.missing_value
        qcf[4:,0] = bufr.missing_value
        qcf[4,0] = 31.
        oer[:,0] = bufr.missing_value
        oer[3,0] = 4

        rh = data["relative_humidity@ObsValue"][i]
        tmp = data["air_temperature@ObsValue"][i]
        if tmp != BAD:
            tmp -= 273.15
        if pr != BAD and tmp != BAD and rh != BAD:
            rdgas = 287.04
            rvgas = 461.50
            eps = rdgas/rvgas

            es = gfssvp(tmp) * min(1, max(0, rh/100. * svpw(tmp)/svp(tmp)))
            qs = eps * es / (pr - (1-eps) * es)
            newspec = qs

            spec = newspec

            rh = spec * 1e6
        else:
            rh = BAD
        if rh != BAD:
            obs[1,0] = rh
            oer[1,0] = data["relative_humidity@ObsError"][i] * 0.7
            qcf[1,0] = 2.
        else:
            qcf[1,0] = 31.
        if tmp != BAD:
            obs[2,0] = tmp
            oer[2,0] = data["air_temperature@ObsError"][i]
            qcf[2,0] = 1.
        else:
            qcf[2,0] = 31.


        bufr.write_subset(hdr,hdstr)
        bufr.write_subset(obs,obstr)
        bufr.write_subset(dr,drstr)
        bufr.write_subset(oer,oestr)
        bufr.write_subset(qcf,qcstr,end=True)

        bufr.close_message()


    bufr.close()
