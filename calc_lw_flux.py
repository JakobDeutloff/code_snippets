# %%
import pyarts
import numpy as np
import xarray as xr
import FluxSimulator as fsm
import zarr

# %%
def calc_spectral_fluxes_from_profiles(
    atmosphere,
    species,
    gases,
    surface_altitude=0.0,
    nstreams=10,
    fnum=300,
    fmin=1.0,
    fmax=75e12,
    verbosity=0,
):
    """Calculate spectral fluxes at top of the atmosphere (TOA) and surface for a given atmosphere profiles.

    Parameters:
    -----------
    atmosphere : xarray.Dataset
        Atmospheric profiles containing pressure, temperature, and gas concentrations.
        Structure: coords=[pressure, lat, lon], variables=[temperature, gases]
    species : list
        List of species to be included in the radiative transfer calculations.
    gases : list
        List of gases to be included in the radiative transfer calculations.

    Returns:
    --------
    rad_field : np.ndarray
    """

    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    ws.abs_speciesSet(
        species=species,
    )

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    if ("H2O-SelfContCKDMT400" in species) or ("H2O-ForeignContCKDMT400" in species):
        ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # Read cross section data
    ws.ReadXsecData(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Calculate absorption
    ws.propmat_clearsky_agendaAuto()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    # This needs to be a loop over lat lon
    lat = atmosphere["lat"][0]
    lon = atmosphere["lon"][0]

    # extract profiles
    pressure = atmosphere.sel(lat=lat, lon=lon)["pressure"].values
    temperature = atmosphere.sel(lat=lat, lon=lon)["temperature"].values

    # Atmosphere and surface
    ws.Touch(ws.lat_grid)
    ws.Touch(ws.lon_grid)
    ws.lat_true = np.array([0.0])
    ws.lon_true = np.array([0.0])

    ws.AtmosphereSet1D()
    ws.p_grid = pressure
    ws.t_field = temperature[:, np.newaxis, np.newaxis]

    # How does arts know which gas is at which pos of vmr?
    vmr_field = np.zeros((len(gases), len(atmosphere["pressure"].values), 1, 1))

    # here we need some loop over gases to fill vmr_field
    vmr_field[0, :, 0, 0] = atmosphere["h2o"].values
    vmr_field[1, :, 0, 0] = atmosphere["co2"].values
    vmr_field[2, :, 0, 0] = atmosphere["ch4"].values
    vmr_field[3, :, 0, 0] = atmosphere["o2"].values
    vmr_field[4, :, 0, 0] = atmosphere["n2"].values
    vmr_field[5, :, 0, 0] = atmosphere["o3"].values
    ws.vmr_field = vmr_field

    ws.z_surface = np.array([[surface_altitude]])
    ws.p_hse = 100000
    ws.z_hse_accuracy = 100.0
    ws.z_field = 16e3 * (
        5 - np.log10(atmosphere["pressure"].values[:, np.newaxis, np.newaxis])
    )
    ws.atmfields_checkedCalc()
    ws.z_fieldFromHSE()

    # Set surface temperature equal to the lowest atmosphere level
    ws.surface_skin_t = ws.t_field.value[0, 0, 0]

    # Output radiance not converted
    ws.StringSet(ws.iy_unit, "1")

    # set cloudbox to full atmosphere
    ws.cloudboxSetFullAtm()

    # set particle scattering to zero, because we want only clear sky
    ws.scat_data_checked = 1
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()

    # No sensor properties
    ws.sensorOff()

    # No jacobian calculations
    ws.jacobianOff()

    # Check model atmosphere
    ws.scat_data_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.lbl_checkedCalc()

    # Perform RT calculations
    ws.spectral_irradiance_fieldDisort(nstreams=nstreams, emission=1)

    # Extract fluxes and build xarray dataset
    rad_field = ws.spectral_irradiance_field

    return rad_field


# %% call function
species = [
    "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
    "CO2, CO2-CKDMT252",
    "CH4",
    "O2,O2-CIAfunCKDMT100",
    "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
    "O3",
]
atmosphere = xr.open_dataset("/Users/jakobdeutloff/Desktop/atms.nc").rename(
    {"geometric height": "geometric_height"}
)
gases = ["H2O", "CO2", "CH4", "O2", "N2", "O3"]
atmosphere = atmosphere.sel(lat=slice(-30, 30), lon=slice(-10, 10))


# %% convert atms to gridded field
atms_grd = pyarts.arts.ArrayOfGriddedField4()

for i, lat_i in enumerate(atmosphere.lat):
    for j, lon_j in enumerate(atmosphere.lon):
        profile = atmosphere.sel(lat=lat_i, lon=lon_j)
        ##add here
        profile_grd = fsm.generate_gridded_field_from_profiles(
            profile["pressure"].values,
            profile["temperature"].values,
            gases={"H2O": profile['H2O'], "CO2": profile['CO2'], 'O3': profile['O3'], 'N2': profile['N2'], 'O2': profile['O2']},
        )
        atms_grd.append(profile_grd)

#%% setup ARTS
f_grid = np.linspace(1, 5e3, 200)
f_grid_freq = pyarts.arts.convert.kaycm2freq(f_grid)
surface_reflectivity_lw = 0.05

LW_flux_simulator = fsm.FluxSimulator("test_atms")
LW_flux_simulator.ws.f_grid = f_grid_freq
LW_flux_simulator.set_species(
    [
        "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
        "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
        "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        "CO2, CO2-CKDMT252",
        "O3",
        "O3-XFIT",
    ]
)

# %% get lookup table
LW_flux_simulator.get_lookuptableBatch(atms_grd)


# %%
fluxes_spectral = xr.Dataset(
    {
        "flux_upward": (("lat", "lon", "pressure", "f_grid"), np.zeros((len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure), len(f_grid)))),
        "flux_downward": (("lat", "lon", "pressure", "f_grid"), np.zeros((len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure), len(f_grid)))),
    },
    coords={"lat": atmosphere.lat, "lon": atmosphere.lon, "f_grid": f_grid},
)

fluxes_integrated = xr.Dataset(
    {
        "flux_upward": (("lat", "lon", "pressure"), np.zeros((len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure)))),
        "flux_downward": (("lat", "lon", "pressure"), np.zeros((len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure)))),
        "heating_rate": (("lat", "lon", "pressure"), np.zeros((len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure)))),
    },
    coords={"lat": atmosphere.lat, "lon": atmosphere.lon},
)

# %%
for lat in atmosphere.lat:
    for lon in atmosphere.lon:
        results_lw = LW_flux_simulator.flux_simulator_single_profile(
            atms_grd[0],
            atmosphere.sel(lat=lat, lon=lon).isel(pressure=0)["temperature"].values,
            np.max([-318, atmosphere.sel(lat=lat, lon=lon).isel(pressure=0)["geometric_height"].values]),
            surface_reflectivity_lw,
            geographical_position=[lat, lon],
        )

        fluxes_spectral["flux_upward"].loc[lat, lon] = results_lw["spectral_flux_clearsky_up"].T
        fluxes_spectral["flux_downward"].loc[lat, lon] = results_lw["spectral_flux_clearsky_down"].T
        fluxes_integrated["flux_upward"].loc[lat, lon] = results_lw["flux_clearsky_up"]
        fluxes_integrated["flux_downward"].loc[lat, lon] = results_lw["flux_clearsky_down"]
        fluxes_integrated["heating_rate"].loc[lat, lon] = results_lw["heating_rate_clearsky"].T
   



# %%
