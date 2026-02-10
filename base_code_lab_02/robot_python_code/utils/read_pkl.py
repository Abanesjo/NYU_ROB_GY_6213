import pickle

with open("/Users/stavanchristian/Documents/PhD/Sem 4/Rl/NYU_ROB_GY_6213/base_code_lab_02/robot_python_code/data/robot_data_50_0_06_02_26_16_00_04.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
