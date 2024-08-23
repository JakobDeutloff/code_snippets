# %%
import time
import xarray as xr 
from pandas import Timedelta
from orcestra.postprocess.level0 import radiometer
from orcestra.postprocess.level1 import resample_radiometer, filter_radiometer

# %% read radiometer data 
ds_kv = xr.open_dataset('/Volumes/ORCESTRA/HALO-20240816a/radiometer/KV/240816.BRT.NC').pipe(radiometer)

# %% Pandas resample 
start_time = time.time()
ds_kv_pd = resample_radiometer(ds_kv, Timedelta('1s'))
print("Pandas resample time: %s seconds" % (time.time() - start_time))

# %% Xarray reasmpe
start_time = time.time()
ds_kv_xr = ds_kv.resample(time='1s').mean()
print("Xarray resample time %s seconds" % (time.time() - start_time))

# %%
