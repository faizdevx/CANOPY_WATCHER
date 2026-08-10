# CANOPY_WATCHER
Computer Aided Network for Observing and Protecting Yielding wildlife habitats through Wireless AI-based Tracking and Camera Health monitoring



# instructions


Same codebase has to work on all three platforms without modifications. Access to cameras, access to GPUs, reliable disk storage and reliable networking is not the same on all three. The challenge is to have the same codebase on all three and no code branches for different hardware.

Traversed the entire system and classified each device-specific decision into seven axes. Lack of any one axis will ultimately lead to the embedding of a device test inside the application code.

| **Axis**                      | **Reason why it cannot be hardcoded**                                                                                                                                                       |
| ------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Identity**                  | node_id/org/site/zone - mandatory for multi-device fleet support, unknown until provisioning                                                                                                    |
| **Hardware**                  | camera/gpio/power driver + `hardware_tier` - unique per device, and Pi (CPU-only) is more constrained than Jetson even though both are SBC                                                         |
| **AI capability**             | accelerator (cpu/cuda/tensorrt/edgetpu), model paths/formats, Tier2 policy - Pi cannot perform Tier2 real-time, whereas Jetson can                                                               |
| **Storage**                   | write batching, retention policy - SD cards will wear down from persistent writes, but laptop SSD does not mind                                                                                   |
| **Networking**                | broker type, synchronization protocol, backoff curve - laptop is online always, Pi relies on flaky WiFi connectivity, Jetson in field expects LoRa/4G                                                 |
| **Health monitoring**         | what sensors are available at all (battery/thermal) - Pi without UPS does not have a battery sensor, whereas Jetson has thermal throttling in Tier2                                                |
| **Debug ergonomics**          | log verbosity, local API, simulation mode - only laptop can have permissive/verbose defaults                                                                                                   |





Critical decision: facts versus policy. The script can determine that there is a GPU. However, it cannot determine if Tier2 should be local, delegated to the cloud, or turned off – that would be a matter of policy. This difference formed the basis for all of the following layering.


3. Layering architecture


```
base.yaml                  common defaults, always present
    ↓
<profile>.yaml              project's behavioral decisions (laptop / rpi / jetson_nano / jetson_orin)
    ↓
hardware.generated.yaml     machine-detected FACTS (auto-written, optional)
    ↓
env vars                    secrets (via *_env indirection) + ad-hoc overrides

```

Why this order specifically:

base.yaml first so that every profile starts from the same safe default, and new entries in the later files don't require editing every single profile

.yaml next because it encodes all and only the information that requires human judgement (Tier2 mode, retention policy, log verbosity)

hardware.generated.yaml after that so that any hardware information discovered on the actual machine takes precedence over what might have been guessed for a similar machine

and finally env vars, because secrets (and node_id) should never be stored in any committed file, and last-layer wins is the only reliable way to ensure that.

The reason config files don't encode any secret values, only env keys (like auth_token_env: CW_SYNC_TOKEN) is that every file.yaml is under version control, and therefore unsafe to store secrets (or node_id!) in.

The loader will error out immediately when it encounters a missing field that it's not prepared to handle, rather than silently loading an incorrect default that would hide a misconfiguration elsewhere.


file structure 

```
canopy-watch/
├── config/                      # DATA — yaml only, no code, safe to commit (no secrets)
│   ├── base.yaml
│   ├── laptop.yaml
│   ├── rpi.yaml
│   ├── jetson_nano.yaml
│   ├── jetson_orin.yaml
│   └── hardware.generated.yaml  # auto-written, gitignored, machine-specific
│
├── edge/
│   ├── config/
│   │   └── loader.py             # merges the 4 layers, resolves secrets, validates
│   ├── capture/ inference/ store/ sync/ ...
│
└── scripts/
    └── detect_hardware.py        # probes OS/CPU/GPU/RAM/camera/storage/network/thermal/battery

```


# Flow (personas)

## contributor


```
git clone <repo> && cd canopy-watch
pip install -r requirements.txt
python scripts/detect_hardware.py            # writes config/hardware.generated.yaml
CW_PROFILE=laptop python -m edge.config.loader   # sanity-check the merged config
docker-compose -f infra/compose/docker-compose.yaml up
```

The use of a mock/webcam driver, in process or local-docker broker, with both Tier1 and Tier2 running locally (inexpensive to test the whole chain), verbose logging, and simulation mode (so that no actual hardware is needed; tests/fixtures simulate the camera feed).

## Raspberry Pi field user

```
ssh pi@<device>
git clone <repo> && cd canopy-watch
pip install -r requirements.txt
python3 scripts/detect_hardware.py
export CW_NODE_ID=rpi-field-03
export CW_PROFILE=rpi
export CW_SYNC_TOKEN=<provisioned token>
python3 -m edge.config.loader     # confirm it validates before deploying as a service
```

Detector correctly sets hardware_tier: constrained and tier2.mode: defer_to_cloud (Pi CPU isn't powerful enough to run Tier2 in real-time, so it's sent to the cloud for verification instead of silently downgrading). Sync uses MQTT with a long backoff maximum to handle being offline for extended periods of time; storage is batched to increase the life of the SD card.

## Jetson Nano/Orin field engineer

```
ssh jetson@<device>
git clone <repo> && cd canopy-watch
pip install -r requirements.txt
python3 scripts/detect_hardware.py    # reads /etc/nv_tegra_release, sets accelerator: tensorrt
export CW_NODE_ID=jetson-orin-north-ridge
export CW_PROFILE=jetson_orin          # or jetson_nano
python3 -m edge.config.loader
```

The detector has been configured to run on Jetson by setting hardware_tier: accelerated, accelerator: tensorrt and CSI camera driver has been installed automatically by reading /etc/nv_tegra_release. The profile enables Tier2 on the Jetson locally and turns on thermal monitoring (Jetson has tendency to throttle when running continously heavy inference). Runs as a systemd service (infra/jetson/) rather than inside docker on the device in the field.


# operation notes

The file detect_hardware.py should be rerun after any hardware change such as adding a new camera, swapping out an SD card, or installing a GPU. The file in question is autogenerated and should be reloaded every time - not edited manually.
hardware.generated.yaml is gitignored - it's specific to a machine, gets re-generated every time detect_hardware is run, and there's no point in trying to merge different versions.
You can override any config value either via simple env vars (for fields common to all devices, like CW_NODE_ID, CW_ORG_ID, CW_SITE, CW_ZONE) or CW__section__key=value format for everything else (example: CW__networking__sync__endpoint=mqtt://gateway.local:1883)
Loader is a library, not just a cli tool - the actual app code always calls get_config(), which in turn reads hardware.generated.yaml. This is what makes drivers, vision models, and sync code all run exactly the same on all three envs.
Validation is designed to be "fail fast": if someone tries to start up a field device without specifying a node_id, it should refuse to start up - rather than default to some null value and proceed to do weird things in that state.


# so what manually you have to do 

**Laptop (contributor)**
- Install deps: `pip install pyyaml psutil`
- Run: `python scripts/detect_hardware.py`
- Manually set: `CW_NODE_ID` (anything, e.g. `dev-yourname-01`)
- macOS/Windows only: script can't auto-find your webcam — manually edit `hardware.generated.yaml`, set `driver.camera: webcam`, `camera_params.device_index: 0`
- Done. Everything else (broker, storage, Tier2) comes from `laptop.yaml`.

**Raspberry Pi**
- Physically: plug in the camera module (or USB webcam), boot the Pi
- If using the Pi Camera Module: `sudo raspi-config` → enable camera interface, reboot
- Install deps: `pip install pyyaml psutil`
- Run: `python3 scripts/detect_hardware.py`
- Manually set:
  - `CW_NODE_ID` (unique per device — you decide this, e.g. `rpi-northgate-01`)
  - `CW_SYNC_TOKEN` (get this from whoever provisions the fleet)
  - `hardware.generated.yaml → driver.power` — leave `null` unless you physically installed a UPS/battery HAT, then set it manually
- Sync endpoint (`networking.sync.endpoint`) — someone gives you this, set via `CW__networking__sync__endpoint=mqtt://...`

**Jetson Nano/Orin**
- Physically: connect CSI camera ribbon cable correctly (this is the #1 failure point — reversed cable = no image, no error)
- Must be flashed with JetPack (gives you `/etc/nv_tegra_release`, which the script needs to detect it's a Jetson at all — no JetPack, script treats it as a generic Linux box)
- Install deps: `pip install pyyaml psutil`
- Run: `python3 scripts/detect_hardware.py`
- Manually set:
  - `CW_NODE_ID`
  - `CW_PROFILE=jetson_nano` or `jetson_orin` (script doesn't pick this — you know which board you're on)
  - `ota.model_registry_endpoint` in the profile yaml (not auto-detected)
  - Sync endpoint, same as Pi

**One command everyone runs to check they did it right:**
```
python -m edge.config.loader --profile <yours>
```
If it errors, it tells you exactly what's still missing.

# follow ups 

requirements.txt / dependency pinning for pyyaml, psutil
.gitignore entry for config/hardware.generated.yaml
macOS/Windows camera enumeration in detect_hardware.py (currently only Linux implemented; falls back to manual override elsewhere)
Fleet-level config push (as per the original design’s note in the “Edge fleet management” scalability subsection) – not applicable to the single-node loader.

