import os
import h5py
import pandapower as pp

# ================= CONFIGURATION =================
# Change these values directly to adjust directories and thresholds
DATA_DIR = "/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/result4.powerflow/"
THRESHOLD_METERS = 13.0
IMPEDANCE_THRESHOLD_OHMS = 0.001
# =================================================

# Convert threshold from meters to kilometers (as pandapower uses km)
threshold_km = THRESHOLD_METERS / 1000.0

# Print the Excel-pasteable table header (separated by tab)
print("File_ID;Short_Lines_Count;Low_Impedance_Lines_Count")

# Iterate over all files in the folder in alphabetical order
for filename in sorted(os.listdir(DATA_DIR)):
    # Only process HDF5 files
    if not filename.endswith(".h5"):
        continue

    # Extract the ID prefix (e.g. "1359" from "1359_N313...h5")
    file_id = filename.split("_")[0]
    file_path = os.path.join(DATA_DIR, filename)

    try:
        # 1. Open the H5 file and load the network JSON data
        with h5py.File(file_path, "r") as f:
            if "raw_data/net" in f:
                json_data = f["raw_data/net"][()]
            elif "grid_top/net" in f:
                json_data = f["grid_top/net"][()]
            else:
                print(f"{file_id}\tError: Network key not found")
                continue

        # 2. Reconstruct the grid from the JSON data
        net = pp.from_json_string(json_data)

        # 3. Count lines shorter than the threshold
        short_lines = net.line[net.line["length_km"] < threshold_km]
        short_count = len(short_lines)

        # 4. Count lines with low resistance (R) or low reactance (X) in Ohms
        r_ohms = net.line["r_ohm_per_km"] * net.line["length_km"]
        x_ohms = net.line["x_ohm_per_km"] * net.line["length_km"]
        
        # Identify lines below the impedance threshold
        low_imp_lines = net.line[(r_ohms <= IMPEDANCE_THRESHOLD_OHMS) | (x_ohms <= IMPEDANCE_THRESHOLD_OHMS)]
        low_imp_count = len(low_imp_lines)

        # 5. Print results in tab-separated format (copy-pasteable into Excel)
        print(f"{file_id};{short_count};{low_imp_count}")

    except Exception as e:
        print(f"{file_id}\tError\tError: {str(e)}")
