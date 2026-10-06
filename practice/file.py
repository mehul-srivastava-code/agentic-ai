import os
def read_file(filename):
    name = os.path.basename(filename)       # strips any folder path like ../../secret
    if not os.path.exists(name):
        return f"Error: '{name}' not found. Files here: {os.listdir('.')}"
    with open(name, encoding="utf-8") as f:
        return f.read()[:2000]               # cap the size