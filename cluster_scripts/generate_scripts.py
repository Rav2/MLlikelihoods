import os

base_filename = "../likelihood.sh"
template = "python sample.py parameters.yaml"
num_copies = 20
analysis = "1908_08215"

def generate_copies():
	with open(base_filename, "r") as file:
		content = file.read()
	for i in range(1, num_copies+1):
		new_filename = f"likelihood_{analysis}-{i}.sh"
		new_line = f"python sample.py parameters_{analysis}-{i}.yaml"
		new_content = content.replace(template, new_line)
		with open(new_filename, "w") as new_file:
			new_file.write(new_content)
		print(f"Created {new_filename}")

if __name__ == '__main__':
	generate_copies()
