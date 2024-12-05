# %% 
import pyarts 
import numpy as np
import xarray as xr 

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

        pressure_profile (ndarray): Pressure profile [Pa].
        temperature_profile (ndarray): Temperature profile [K].
        h2o_profile (ndarray): Water vapor profile [VMR].
        N2 (float): Nitrogen volume mixing ratio. Defaults to 0.78.
        O2 (float): Oxygen volume mixing ratio. Defaults to 0.21.
        CO2 (float): Carbon dioxide volume mixing ratio. Defaults to 400 ppm.
        CH4 (float): Methane volume mixing ratio. Defaults to 1.8 ppm.
        O3 (float): Ozone volume mixing ratio. Defaults to 0.
        surface_altitude (float): Surface altitude [m]. Defaults to 0.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray: Frequency grid [Hz], spectral fluxes (toa_up, toa_down, sfc_up, sfc_down) [Wm^-2]

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
    if ('H2O-SelfContCKDMT400' in species) or ('H2O-ForeignContCKDMT400' in species):
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

    # This needs to be aloop ver lat lon
    lat = atmosphere['lat'][0]
    lon = atmosphere['lon'][0]

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

    # How does arts know which gas is at which pois of vmr? 
    vmr_field = np.zeros((len(gases), len(atmosphere["pressure"].values), 1, 1))
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
    ws.z_field = 16e3 * (5 - np.log10(atmosphere['pressure'].values[:, np.newaxis, np.newaxis]))
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
atmosphere = xr.open_dataset('/Users/jakobdeutloff/Desktop/atms.nc').rename({'geometric height': 'geometric_height'})
gases = ['H2O', 'CO2', 'CH4', 'O2', 'N2', 'O3']

flux = calc_spectral_fluxes_from_profiles(atmosphere, species, gases)


# %%
