
import re

def parse_c_structures(file_path):
    """
    Parses a C header file to identify and extract structures, and returns a dictionary representation of the structures.

    Args:
    file_path (str): The path to the C header file.

    Returns:
    dict: A dictionary where keys are structure names and values are dictionaries of field names and their types.
    """
    struct_dict = {}
    parent_struct_dcit = {}
    struct_pattern = re.compile(r"^struct\s+(\w+)\s*\{$")
    field_pattern = re.compile(r"^\s*(int|double)\s+(\w+);$")
    child_struct_pattern = re.compile(r"^\s*struct\s+(\w+)\s+(\w+);$")
    end_struct_pattern = re.compile(r"^\};$")
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        current_struct = None
        for line in lines:
            if current_struct:
                # Check if we reached the end of the struct
                if end_struct_pattern.search(line):
                    current_struct = None
                else:
                    # Try to find a field definition
                    field_match = field_pattern.search(line)
                    child_struct_match = child_struct_pattern.search(line)
                    if field_match:
                        dtype, field_name = field_match.groups()
                        struct_dict[struct_name][field_name] = f"ctypes.c_{dtype}"
                    elif child_struct_match:
                        child_struct_type, child_struct_name = child_struct_match.groups()
                        struct_dict[struct_name][child_struct_name] = child_struct_type
            else:
                # Check for a new struct definition
                struct_match = struct_pattern.search(line)
                if struct_match:
                    struct_name = struct_match.group(1)
                    struct_dict[struct_name] = {}
                    current_struct = struct_name

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return struct_dict


def generate_python_structs(struct_dict):
    """
    Generates Python ctypes structure definitions based on the dictionary representation of C structures.

    Args:
    struct_dict (dict): A dictionary where keys are structure names and values are dictionaries of field names and their types.

    Returns:
    str: A string containing the Python ctypes structure definitions.
    """
    struct_definitions = []
    for struct_name, fields in struct_dict.items():
        struct_definition = f"class {struct_name}(ctypes.Structure):\n"
        struct_definition += "    _fields_ = [\n"
        for field_name, field_type in fields.items():
            struct_definition += f"        (\"{field_name}\", {field_type}),\n"
        struct_definition += "    ]\n\n"
        struct_definitions.append(struct_definition)
    
    return "\n".join(struct_definitions)

def generate_py_file(file_path, targetpath):
    """
    Generates a Python file containing ctypes structure definitions based on the C header file.

    Args:
    file_path (str): The path to the C header file.

    Returns:
    str: A string containing the Python ctypes structure definitions.
    """
    struct_dict = parse_c_structures(file_path)
    python_structs = generate_python_structs(struct_dict)
    with open(targetpath, 'w') as file:
        file.write(python_structs)
    
    print(f"Python ctypes structures generated and saved to {targetpath}")
    return None

if __name__ == "__main__":
    generate_py_file(r"C:\Users\vanei\source\repos\Genetic Algorithm - C Branch\src\Helper\Struct.h", r"C:\Users\vanei\source\repos\Genetic Algorithm - C Branch\src\x64\DLL Build\Structs.py")