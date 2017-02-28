
import pandas as pd
import shutil
import requests

with open('../opendatakey.txt') as keyfile:
	apikey = keyfile.read().rstrip()

datetimes = pd.date_range('2017-02-25', '2017-02-28', freq='5min')

for dtime in datetimes:

	timestr = dtime.strftime('%Y-%m-%dT%H:%M:%SZ')
	request = 'http://wms.fmi.fi/fmi-apikey/{apikey:s}/geoserver/Radar/wms?service=WMS&version=1.3.0&request=GetMap&layers=Radar:suomi_dbz_eureffin&styles=raster&bbox=100000,6300000,600000,7500000&srs=EPSG:3067&format=image%2Fgeotiff&time={time:s}&width=200&height=200'.format(apikey=apikey, time=timestr)
	print(request)

	r = requests.get(request, stream=True)
	if r.status_code == 200:
	    with open('../data/%s.tiff'%timestr, 'wb') as f:
	        shutil.copyfileobj(r.raw, f)