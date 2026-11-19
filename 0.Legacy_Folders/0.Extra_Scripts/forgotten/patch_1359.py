import os
import pandas as pd

# Resolve absolute path to the target H5 file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILEPATH = os.path.join(CURRENT_DIR, '3.urbs', 'Input', '1359_N3133500E4491500_92224_5_12.h5')

print(f"Patching: {FILEPATH}")

with pd.HDFStore(FILEPATH, mode='a') as store:
    # Patch EV Mobility Storage (mobility_storage0) at Site 154
    if '/urbs_in/storage' in store:
        sto = store['/urbs_in/storage']
        mask = (sto['Site'].astype(str) == '154') & (sto['Storage'] == 'mobility_storage0')
        if mask.any():
            original_cap_up = sto.loc[mask, 'cap-up-c'].values[0]
            original_inst_cap = sto.loc[mask, 'inst-cap-c'].values[0]
            print(f"  [Original] 'mobility_storage0' at Site 154: cap-up-c = {original_cap_up} kWh, inst-cap-c = {original_inst_cap} kWh")
        sto.loc[mask, 'cap-up-c'] = 120.0
        #sto.loc[mask, 'inst-cap-c'] = 120.0
        store['/urbs_in/storage'] = sto #Writes sto back into the .h5 file: saves the modification
        print("  Updated 'mobility_storage0' capacity at Site 154 to 120.0 kWh")

print("Patching complete!")
