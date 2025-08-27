import os
import shutil

# Base filename and range of copies
analysis = "1908.08215"
data_path = f"../data/{analysis}"
base_filename = f"parameters_{analysis.replace('.','_')}-1.yaml"
template_include = f"{analysis}-1"
num_copies = 20

def generate_yaml_copies():
    with open(base_filename, "r") as file:
        content = file.read()
    
    with open(f"parameter_cards/{analysis}.yaml", "r") as fparam:
        fparam_content = fparam.read()

    for i in range(1, num_copies + 1):
        #new_filename = f"parameters_1908_08215-{i}.yaml"
        new_filename = f"parameters_{analysis.replace('.','_')}-{i}.yaml"
        new_include = f"{analysis}-{i}"
        new_content = content.replace(template_include, new_include)
        
        with open(new_filename, "w") as new_file:
            new_file.write(new_content)
	
        with open(f"parameter_cards/{analysis}-{i}.yaml", "w") as fout:
            fout.write(fparam_content)
        
        print(f"Created {new_filename}")

        shutil.copytree(data_path, data_path+f'-{i}')
        print(f"Copied data to {data_path}-{i}")
	

if __name__ == "__main__":
    generate_yaml_copies()

