class Config():
    ### Data directories
    #DATA_DIR = "Input/"
    #STORAGE_DIR = "Output/"
    DATA_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/2nd_RUN/3.Post_Urbs/"
    STORAGE_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/2nd_RUN/4.Power_Flown/"

    ### Power factors
    PF_PV_MIN = 0.95    # Photovoltaics: -arccos(PF_MIN) <= tan(phi)=Q/P <= arccos(PF_MIN), source: VDE-AR-N 4105
    PF_HP = 0.95        # Heat pump,     source: example data sheet - https://www.solarwatt.de/canto/download/bnu1pcavot0oh2bem4qpi4k63i
    PF_ELC = 0.959      # Electricity,   source: https://www.researchgate.net/publication/285577915_Representative_electrical_load_profiles_of_residential_buildings_in_Germany_with_a_temporal_resolution_of_one_second

config = Config()