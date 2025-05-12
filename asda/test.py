import subprocess

def print_python_versions():
    # 获取主环境中的 Python 版本
    print("Python version in the main environment:")
    result_main_env = subprocess.run(
        ['python', '--version'],  # 获取主环境中的 Python 版本
        capture_output=True,
        text=True
    )
    print(result_main_env.stdout)

    # 获取 data 环境中的 Python 版本
    print("\nPython version in the 'data' environment:")
    result_data_env = subprocess.run(
        ['conda', 'run', '-n', 'data', 'python', '--version'],  # 获取 data 环境中的 Python 版本
        capture_output=True,
        text=True
    )
    print(result_data_env.stdout)

# 调用函数打印版本
if __name__ == '__main__':
    print_python_versions()
