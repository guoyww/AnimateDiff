import os

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_absolute_path(*relative):
    if relative[0].startswith('/'):
        return os.path.join(*relative)  # absolute path
    return os.path.join(project_path, *relative)


if __name__ == '__main__':
    print(get_absolute_path('test'))
    print(get_absolute_path('/test'))
    print(get_absolute_path('test', 'test'))
    print(get_absolute_path('/test', 'test'))
