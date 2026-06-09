import os

def get_project_root():
    current=os.path.dirname(os.path.abspath(__file__))
    root_path=os.path.dirname(current)
    return root_path

def get_abs_path(relative_path):
    root_path=get_project_root()
    abs_path=os.path.join(root_path,relative_path)
    return abs_path

if __name__ == '__main__':
    print(get_abs_path("config.json"))