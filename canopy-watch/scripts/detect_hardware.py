#!/usr/bin/env python3
"""
detect_hardware.py — Canopy Watch hardware detection

Probes the local machine for hardware FACTS (OS, CPU, GPU, RAM, cameras,
storage, network, temperature/battery capability) and writes them to
config/hardware.generated.yaml.

This script determines FACTS only. It never decides POLICY — e.g. whether
Tier2 should run locally, retention limits, or sync endpoints. Those stay
in base.yaml / <profile>.yaml. The computer isn't clairvoyant.

Layering (config loader applies in this order, later wins):
    base.yaml -> <profile>.yaml -> hardware.generated.yaml -> env vars

Usage:
    python scripts/detect_hardware.py
    python scripts/detect_hardware.py --output config/hardware.generated.yaml

  uv add psutil
  
  into env 
"""

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------- low-level probes ----------

def run(cmd):
    """Run a command, return stdout or None on failure/absence."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_os():
    return {
        "system": platform.system(),      # Linux / Darwin / Windows
        "release": platform.release(),
        "machine": platform.machine(),    # x86_64 / aarch64 / arm64 ...
    }


def detect_jetson():
    """Returns Jetson model string if this is a Jetson device, else None."""
    tegra_release = Path("/etc/nv_tegra_release")
    model_path = Path("/proc/device-tree/model")
    if tegra_release.exists():
        if model_path.exists():
            return model_path.read_text(errors="ignore").strip("\x00").strip()
        return "Jetson (model unknown)"
    return None


def detect_raspberry_pi():
    """Returns Pi model string if this is a Raspberry Pi, else None."""
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        model = model_path.read_text(errors="ignore").strip("\x00").strip()
        if "Raspberry Pi" in model:
            return model
    return None


def detect_cpu():
    return {
        "arch": platform.machine(),
        "logical_cores": os.cpu_count() or 1,
        "model": platform.processor() or _cpu_model_fallback(),
    }


def _cpu_model_fallback():
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return "unknown"


def detect_gpu(jetson_model):
    """Returns (accelerator, gpu_name)."""
    if jetson_model:
        return "tensorrt", jetson_model
    nvidia_smi = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if nvidia_smi:
        return "cuda", nvidia_smi.splitlines()[0]
    return "cpu", None


def detect_ram_mb():
    if HAS_PSUTIL:
        return round(psutil.virtual_memory().total / (1024 * 1024))
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return round(int(line.split()[1]) / 1024)
        except OSError:
            pass
    return None


def detect_cameras():
    """Returns list of detected Linux video device paths. Empty elsewhere —
    macOS/Windows enumeration needs extra deps, left as a manual override."""
    if platform.system() == "Linux":
        return sorted(str(p) for p in Path("/dev").glob("video*"))
    return []


def detect_libcamera():
    """Detects the libcamera stack used by newer Pi camera modules,
    which may not expose a /dev/video* node."""
    return run(["which", "libcamera-hello"]) is not None


def detect_storage(path="/"):
    total, _used, free = shutil.disk_usage(path)
    return {
        "total_mb": round(total / (1024 * 1024)),
        "free_mb": round(free / (1024 * 1024)),
    }


def detect_network():
    interfaces = []
    if HAS_PSUTIL:
        interfaces = [n for n in psutil.net_if_addrs().keys() if n != "lo"]
    online = False
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        online = True
    except OSError:
        online = False
    return {"interfaces": interfaces, "online": online}


def detect_battery():
    if HAS_PSUTIL:
        batt = psutil.sensors_battery()
        return batt is not None
    power_supply = Path("/sys/class/power_supply")
    if power_supply.exists():
        return any("BAT" in p.name for p in power_supply.iterdir())
    return False


def detect_thermal():
    if HAS_PSUTIL and hasattr(psutil, "sensors_temperatures"):
        return bool(psutil.sensors_temperatures())
    return Path("/sys/class/thermal/thermal_zone0/temp").exists()


# ---------- classification: facts -> schema-compatible values ----------

def classify(gpu_accelerator, ram_mb, jetson_model, pi_model, cameras, has_libcamera):
    if jetson_model:
        hardware_tier = "accelerated"
        has_csi = bool(cameras) or has_libcamera
        driver = {
            "camera": "csi_gstreamer" if has_csi else "mock",
            "camera_params": {"sensor_id": 0} if has_csi else {},
            "gpio": "jetson_gpio",
            "power": "jetson_power",
        }
    elif pi_model:
        hardware_tier = "constrained"
        has_cam = bool(cameras) or has_libcamera
        driver = {
            "camera": "picamera" if has_cam else "mock",
            "camera_params": {"camera_index": 0} if has_cam else {},
            "gpio": "rpi_gpio",
            # None: only set if a UPS/battery HAT is manually confirmed present
            "power": None,
        }
    else:
        # laptop / generic workstation
        hardware_tier = "accelerated" if gpu_accelerator == "cuda" and ram_mb and ram_mb >= 8192 else "constrained"
        driver = {
            "camera": "webcam" if cameras else "mock",
            "camera_params": {"device_path": cameras[0]} if cameras else {},
            "gpio": "mock",
            "power": "mock",
        }
    return {"hardware_tier": hardware_tier, "driver": driver}


# ---------- output ----------

def build_yaml(os_info, cpu_info, gpu_accelerator, gpu_name, ram_mb,
                storage_info, network_info, has_battery, has_thermal, hw_facts):
    L = []
    L.append("# hardware.generated.yaml")
    L.append("# AUTO-GENERATED by scripts/detect_hardware.py — do not hand-edit, re-run instead.")
    L.append(f"# Detected on: {os_info['system']} {os_info['release']} ({os_info['machine']})")
    L.append("#")
    L.append("# States FACTS about this machine only (what hardware exists).")
    L.append("# Does not set POLICY (tier2.mode, retention limits, sync endpoints, etc.) —")
    L.append("# those stay in base.yaml / <profile>.yaml.")
    L.append("#")
    L.append("# Layer order: base.yaml -> <profile>.yaml -> hardware.generated.yaml -> env vars")
    L.append("#")
    L.append("# detected_facts (informational, not consumed by the config loader):")
    L.append(f"#   cpu: {cpu_info['model']} ({cpu_info['logical_cores']} cores, {cpu_info['arch']})")
    L.append(f"#   gpu: {gpu_name or 'none detected'}")
    L.append(f"#   ram_mb: {ram_mb if ram_mb is not None else 'unknown'}")
    L.append(f"#   storage_free_mb: {storage_info['free_mb']} / {storage_info['total_mb']}")
    L.append(f"#   network_interfaces: {', '.join(network_info['interfaces']) or 'none'}")
    L.append(f"#   network_online_at_detect_time: {network_info['online']}")
    L.append("")
    L.append("hardware:")
    L.append(f"  hardware_tier: {hw_facts['hardware_tier']}")
    L.append("  driver:")
    L.append(f"    camera: {hw_facts['driver']['camera']}")
    params = hw_facts['driver']['camera_params']
    if params:
        L.append("    camera_params:")
        for k, v in params.items():
            L.append(f"      {k}: {v}")
    else:
        L.append("    camera_params: {}")
    L.append(f"    gpio: {hw_facts['driver']['gpio']}")
    power = hw_facts['driver']['power']
    if power is None:
        L.append("    power: null  # set manually if a UPS/battery HAT is installed")
    else:
        L.append(f"    power: {power}")
    L.append("")
    L.append("ai:")
    L.append(f"  accelerator: {gpu_accelerator}")
    L.append("")
    L.append("health:")
    L.append(f"  battery_monitor: {str(has_battery).lower()}")
    L.append(f"  thermal_monitor: {str(has_thermal).lower()}")
    L.append("  disk_monitor: true")
    L.append(f"  connectivity_monitor: {str(len(network_info['interfaces']) > 0).lower()}")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Detect local hardware, generate hardware.generated.yaml")
    parser.add_argument("--output", default="config/hardware.generated.yaml",
                         help="Output path (default: config/hardware.generated.yaml)")
    args = parser.parse_args()

    if not HAS_PSUTIL:
        print("note: psutil not installed — falling back to platform-specific probes for "
              "RAM/network/battery/thermal. `pip install psutil` for more reliable detection.",
              file=sys.stderr)

    os_info = detect_os()
    jetson_model = detect_jetson()
    pi_model = detect_raspberry_pi()
    cpu_info = detect_cpu()
    accelerator, gpu_name = detect_gpu(jetson_model)
    ram_mb = detect_ram_mb()
    cameras = detect_cameras()
    has_libcamera = detect_libcamera()
    storage_info = detect_storage()
    network_info = detect_network()
    has_battery = detect_battery()
    has_thermal = detect_thermal()

    hw_facts = classify(accelerator, ram_mb, jetson_model, pi_model, cameras, has_libcamera)

    yaml_text = build_yaml(os_info, cpu_info, accelerator, gpu_name, ram_mb,
                            storage_info, network_info, has_battery, has_thermal, hw_facts)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text)

    device_label = jetson_model or pi_model or os_info["system"]
    print(f"Detected: {device_label} — tier={hw_facts['hardware_tier']}, "
          f"accelerator={accelerator}, camera={hw_facts['driver']['camera']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()