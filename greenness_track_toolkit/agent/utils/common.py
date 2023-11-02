from greenness_track_toolkit.utils import get_logger


def get_tf_config():
    import os
    tf_config = ''
    try:
        tf_config = os.environ.get("TF_CONFIG", "")
    except RuntimeError as e:
        get_logger().error(f"there is no configuration of TF_CONFIG {e}")
    return tf_config


def get_gpu_visible():
    import pynvml
    from pynvml import NVMLError
    from greenness_track_toolkit.utils import get_logger
    try:
        pynvml.nvmlInit()
        return True
    except NVMLError as e:
        get_logger().error(e)
        return False


def get_host_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        ip = '127.0.0.1'
        get_logger().error(e)
    return ip


def get_computer_info():
    import cpuinfo
    import platform
    from greenness_track_toolkit.agent.models.do import ComputerInfo
    # cpu info
    cpu = cpuinfo.get_cpu_info()
    cpu_brand = cpu['brand_raw']
    cpu_kernel_nums = cpu['count']
    cpu_clock = cpu['hz_advertised_friendly']
    python_version = cpu['python_version']
    os_info = platform.platform()
    has_gpu = get_gpu_visible()
    # gpu info
    if has_gpu:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpu_brands = []
        gpu_clocks = []
        for ind in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(ind)
            gpu_brands.append(f"GPU:{ind} {pynvml.nvmlDeviceGetName(handle)}")
            gpu_clocks.append(f"GPU:{ind} {pynvml.nvmlDeviceGetMemoryInfo(handle).total}")
        gpu_brand = str.join(",", gpu_brands)
        gpu_nums = pynvml.nvmlDeviceGetCount()
        gpu_clock = str.join(",", gpu_clocks)
        pynvml.nvmlShutdown()
    else:
        gpu_brand = ""
        gpu_nums = 0
        gpu_clock = ""
    ip = get_host_ip()
    return ComputerInfo(
        cpu_brand=cpu_brand,
        cpu_kernel_nums=cpu_kernel_nums,
        cpu_clock=cpu_clock,
        gpu_brand=gpu_brand,
        gpu_nums=gpu_nums,
        gpu_clock=gpu_clock,
        agent_ip=ip,
        python_version=python_version,
        os_info=os_info
    )