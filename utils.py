import copy
import os
import subprocess as sub
import gdown
import yaml

def yaml_to_dict(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return copy.deepcopy(data)

def dict_to_yaml(data, filename):
    with open(filename, 'w') as file:
        yaml.safe_dump(data, file)

def download_from_drive_gdown(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def verbose_print(text):
    print(f"{text}")

def create_dirs(arb_path):
    if not os.path.exists(arb_path):
        os.makedirs(arb_path)

def make_all_dirs(list_of_paths):
    for path in list_of_paths:
        create_dirs(path)

def execute(cmd):
    popen = sub.Popen(cmd, stdout=sub.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise sub.CalledProcessError(return_code, cmd)

def is_windows():
    return os.name == 'nt'
