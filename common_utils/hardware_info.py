import psutil
import subprocess
import xml.etree.ElementTree as ET

# ----------------------------------------
# CPU
# ----------------------------------------
def get_cpu_usage():
    """Return CPU usage percentage."""
    return psutil.cpu_percent(interval=0.3)


# ----------------------------------------
# RAM
# ----------------------------------------
def get_ram_usage():
    """Return RAM usage (used, total, percent)."""
    mem = psutil.virtual_memory()
    return {
        "used": mem.used,
        "total": mem.total,
        "percent": mem.percent
    }


# ----------------------------------------
# GPU (Single NVIDIA GPU using nvidia-smi)
# ----------------------------------------
def get_gpu_usage():
    """Return NVIDIA GPU usage info for GPU 0 via nvidia-smi."""
    try:
        # Run nvidia-smi with XML output
        result = subprocess.run(
            ['nvidia-smi', '-q', '-x', '-i', '0'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return {"vendor": "NVIDIA", "error": "nvidia-smi command failed"}
        
        # Parse XML output
        root = ET.fromstring(result.stdout)
        gpu = root.find('gpu')
        
        if gpu is None:
            return {"vendor": "NVIDIA", "error": "No GPU found"}
        
        # Extract GPU name
        name = gpu.find('product_name')
        name = name.text if name is not None else "Unknown"
        
        # Extract utilization
        utilization = gpu.find('utilization')
        gpu_util = utilization.find('gpu_util') if utilization is not None else None
        util_percent = gpu_util.text.replace('%', '').strip() if gpu_util is not None else "0"
        
        # Extract memory info
        fb_memory = gpu.find('fb_memory_usage')
        mem_used = fb_memory.find('used') if fb_memory is not None else None
        mem_total = fb_memory.find('total') if fb_memory is not None else None
        
        # Parse memory values (format: "X MiB")
        mem_used_mb = int(mem_used.text.split()[0]) if mem_used is not None else 0
        mem_total_mb = int(mem_total.text.split()[0]) if mem_total is not None else 0
        
        return {
            "vendor": "NVIDIA",
            "name": name,
            "util_percent": float(util_percent),
            "memory_used_mb": mem_used_mb,
            "memory_total_mb": mem_total_mb
        }
        
    except FileNotFoundError:
        return {"vendor": None, "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"vendor": "NVIDIA", "error": "nvidia-smi command timed out"}
    except Exception as e:
        return {"vendor": "NVIDIA", "error": f"Error parsing nvidia-smi output: {str(e)}"}