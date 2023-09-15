import os

directory = r'C:\Users\Adam\Documents\Adam\SCHOOL\FinalYear\Thesis\test\temp'  # Change this to your annotations directory path

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r') as file:
            lines = file.readlines()

        with open(os.path.join(directory, filename), 'w') as file:
            for line in lines:
                if line.startswith(
                        "15 "):  # Note the space after 15. This ensures it matches class labels, not part of coordinates.
                    line = line.replace("15 ", "0 ", 1)  # The "1" here ensures only the first occurrence is replaced.
                file.write(line)
