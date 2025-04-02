import pytest
from sedkit import sed
import astropy.units as u
import numpy as np


@pytest.fixture
def sub_spec():
    # Initialize the substellar SED
    sub_spec = sed.SED(name="2MASS J04151954-0935066", substellar=True)
    # Load in a spectrum to test
    sub_spec.add_spectrum_file(
        "tests/data/2MASS_J04151954-0935066_apparent_SED.txt",
        wave_units=u.micron,
        flux_units=u.erg / u.s / u.cm**2 / u.AA,
    )
    return sub_spec


@pytest.fixture
def spec():
    # Initialize the non_substellar SED
    spec = sed.SED(name="2MASS J04151954-0935066", substellar=False)
    # Load in a spectrum to test
    spec.add_spectrum_file(
        "tests/data/2MASS_J04151954-0935066_apparent_SED.txt",
        wave_units=u.micron,
        flux_units=u.erg / u.s / u.cm**2 / u.AA,
    )
    return spec


@pytest.mark.parametrize("seds", ["sub_spec", "spec"])
def test_just_spectrum(seds, request):
    sed = request.getfixturevalue(seds)
    sed.results
    assert np.isclose(sed.fbol[0], 1.9184645e-12 * u.erg / u.s / u.cm**2)
    assert np.isclose(sed.fbol[1], 5.26164965e-15 * u.erg / u.s / u.cm**2)
    assert sed.mbol == (17.811, 0.003)
    assert sed.Lbol is None
    assert sed.radius is None
    assert sed.Teff is None
    assert sed.logg is None
    assert sed.mass is None


@pytest.mark.parametrize(
    "seds,radius_expected,teff_expected",
    [
        (
            "sub_spec",
            (0.973 * u.Rjup, 0 * u.Rjup, 0 * u.Rjup),
            (903 * u.K, 4 * u.K, 4 * u.K),
        ),
        (
            "spec",
            (0.1 * u.solRad, 0 * u.solRad, 0 * u.solRad),
            (903 * u.K, 4.0 * u.K, 4.0 * u.K),
        ),
    ],
)
def test_age_distance(seds, radius_expected, teff_expected, request):
    sed = request.getfixturevalue(seds)
    sed.age = 4.5 * u.Gyr, 0.1 * u.Gyr
    sed.distance = 10 * u.pc, 0.1 * u.pc
    sed.results

    assert np.isclose(sed.fbol[0], 1.9184645e-12 * u.erg / u.s / u.cm**2)
    assert np.isclose(sed.fbol[1], 5.26164965e-15 * u.erg / u.s / u.cm**2)
    assert sed.mbol == (17.811, 0.003)
    assert np.isclose(sed.Lbol[0], 2.29543367e28 * u.erg / u.s, rtol=0.05)
    assert np.isclose(sed.Lbol[1], 4.63561547e26 * u.erg / u.s, rtol=0.05)
    assert sed.Lbol_sun == (-5.222, 0.009)
    assert sed.radius == radius_expected
    assert sed.Teff == teff_expected
    assert sed.logg is None
    assert sed.mass is None


@pytest.mark.parametrize("seds", ["sub_spec", "spec"])
def test_radius(seds, request):
    sed = request.getfixturevalue(seds)
    sed.age = 4.5 * u.Gyr, 0.1 * u.Gyr
    sed.distance = 10 * u.pc, 0.1 * u.pc
    sed.evo_model = "hybrid_solar_age"  # Saumon & Marley 2008 evo model
    sed.results
    sed.infer_radius(infer_from="evo_model")
    if seds == "sub_spec":
        print(sed.radius)
        assert sed.radius == (0.824 * u.Rjup, 0.0 * u.Rjup, 0.0 * u.Rjup)
