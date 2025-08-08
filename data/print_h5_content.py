import h5py
import sys

def list_h5_keys(filename):
    with h5py.File(filename, 'r') as f:
        def visit(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
                #if name == "Jets":
                    # Peek at the first entry
                    #print(f"{indent}  First entry: {obj[0]}")
        f.visititems(visit)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.h5>")
        sys.exit(1)
    list_h5_keys(sys.argv[1])
