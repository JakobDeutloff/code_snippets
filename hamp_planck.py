# %%
import matplotlib.pyplot as plt
import numpy as np


# %% Constants
h = 6.626e-34  # Planck constant
c = 3.0e8  # speed of light
k = 1.38e-23  # Boltzmann constant

# %% 
def planck_function(T, f):
    """
    Planck function for a given temperature and wavelength.
    
    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    wavelen : float
        Wavelength in meters.
        
    Returns
    -------
    float
        Planck function value.
    """
    B = ((2 * h * f**3) / (c**2)) * (1 / (np.exp(h * f / (k * T)) - 1))
    return B
# %% plot planck function for surface temp and cloud temp

frequency = np.linspace(20, 200, 1000)  # frequency in GHz
wavelen = c / (frequency * 1e9)   # wavelength in meters
wavenumber = 1 / (wavelen * 100)  # wavenumber in cm^-1

T_surface = 300 # surface temperature in K
T_cloud = 250  # cloud temperature in K
T_sun = 6000  # sun temperature in K

B_surface = planck_function(T_surface, frequency * 1e9)
B_cloud = planck_function(T_cloud, frequency * 1e9)
B_sun = planck_function(T_sun, frequency * 1e9)

# %% plot 
fig, ax  = plt.subplots()
ax.plot(wavenumber, B_surface, label='Surface')
ax.plot(wavenumber, B_cloud, label='Cloud')
ax.plot(wavenumber, B_sun, label='Sun')
ax.set_xlabel('Wavenumber (cm$^{-1}$)')
ax.set_ylabel('Planck function')
ax.legend()

# %%
