import spiceypy as spice
spice.furnsh("spice_nh/mu69_porter_2024_v01.bds")
spice.spkobj("spice_nh/mu69_porter_2024_v01.bds")
