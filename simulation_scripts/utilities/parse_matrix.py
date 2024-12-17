import re
import numpy as np

def parse_file_to_matrix(filename):
    # Dictionary to store values, using tuples (i, j) as keys
    data = {}
    max_i, max_j = 0, 0

    with open(filename, 'r') as file:
        for line in file:
            # Extract indices and values using regular expressions
            match = re.match(r'a\((\d+),(\d+)\) = ([\d\.\-eE]+)', line.strip())
            if match:
                i, j, value = int(match.group(1)), int(match.group(2)), float(match.group(3))
                data[(i, j)] = value
                max_i, max_j = max(max_i, i), max(max_j, j)

    # Create a matrix of the appropriate size, initialized with zeros
    matrix = np.zeros((max_i + 1, max_j + 1))
    
    # Fill the matrix with the parsed values
    for (i, j), value in data.items():
        matrix[i][j] = value

    return matrix

def read_vector_from_file(filename):
    vector = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line by '=' and strip spaces
            parts = line.split('=')
            if len(parts) == 2:
                # Get the value, strip spaces, and convert to float
                value = float(parts[1].strip())
                vector.append(value)
    return vector

def print_A_and_b(A,b):
    k = 0
    nb = len(b)
    print("\t\t\t\t Matrix \t\t\t\t |RHS|")
    for row in A:
        # row_s = " ".join(f"{val:.0f}" if val != 0 else " " for val in row)
        Ak_s = []
        for val in row:
            if val == 1:
                Ak_s.append("1")
            elif val == 0:
                Ak_s.append(" ")
            elif val>0:
                Ak_s.append("+")
            else:
                Ak_s.append("-")

        Ak_s = " ".join(Ak_s)

        if k < len(b):
            if b[k] == 0:
                bk_s = "0"
            else:
                bk_s = "-"
        else:
            bk_s = " "

        bk_s = "| " + bk_s + " |"

        Ak_s = "| " + Ak_s + " |"
        Ak_s = "%2d: "%k+ Ak_s + bk_s + " :%2d"%k
        print(Ak_s)
        k += 1

# Use the functions
A = parse_file_to_matrix('A.txt')
b = read_vector_from_file('b.txt')
print_A_and_b(A,b)