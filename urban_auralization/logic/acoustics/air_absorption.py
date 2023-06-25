import numpy as np

def air_absorption(freq, relative_humidity, temperature_c, ambient_pressure):
    """
    Air absorbion from ISO 9613
    Parameters
    ----------
    freq : freq vec
    relative_humidity
    temperature_c
    ambient_pressure

    Returns
    -------
    alpha : ndarray
        The attenuation coefficient a, in decibels per metre
        for atmospheric absorption
    """
    t = 273.14 + temperature_c # temp in Kelvin
    t0 = 293.15 #ref temp in Kelvin
    t01 = 273.16 #triple-point isotherm temp in Kelvin (0.01 deg C)
    pa = ambient_pressure
    pr = 101.325e3 #Pa
    c = -6.8346*(t01/t)**1.261+4.6151
    psat = pr*10**c
    h = relative_humidity*psat/pa
    fro = pa/pr*(24+4.04e4*h*(0.02+h)/(0.391+h))
    frn = pa/pr*(t/t0)**(-1/2)*(9+280*h*np.exp(-4.170*((t/t0)**(-1/3)-1)))

    alpha = [8.686 * f**2 * ((1.84e-11*(pa/pr)**(-1)*(t/t0)**(1/2))+
                             (t/t0)**(-5/2)*
                             (0.01275*np.exp(-2239.1/t)*(fro+f**2/fro)**(-1)+
                              0.1068*np.exp(-3352.0/t)*(frn+f**2/frn)**(-1))) for f in freq]


    return np.array(alpha)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pprint
    freq = np.array([50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000])

    a = air_absorption(freq=freq, relative_humidity=10, temperature_c=20, ambient_pressure=101.325e3)
    plt.plot(freq, a)
    plt.show()
    pprint.pprint(a)