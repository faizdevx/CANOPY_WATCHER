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

# CAMERA DRIVER INTERFACE 

### Camera Driver + YAML

The YAML configuration chooses and configures the camera driver.

```text
base.yaml
  ↓
profile.yaml
  ↓
hardware.generated.yaml
  ↓
env vars
  ↓
Final config
  ↓
Camera Factory
  ↓
Camera Driver
  ↓
Standard Frame
  ↓
Capture Service → Tier1/Tier2
```

Example:

```yaml
camera: webcam

camera_params:
  device_name: "Logitech C920"
  device_index: 0
  width: 1280
  height: 720
  fps: 30
```

`camera` chooses the driver:

* `mock` -> mock camera
* `webcam` -> OpenCV webcam
* `picamera` -> Raspberry Pi CSI
* `csi_gstreamer` -> Jetson CSI

The **Config Loader** loads the YAML configuration. The **Factory** converts the chosen driver name into an implementation. The **driver** takes care of hardware-specific capturing and produces a standard `Frame`.

The rest of the code works with `Frame` and does not depend on which hardware we use - laptop, Raspberry Pi or Jetson.

### Why YAML plus Driver Interface?

because

they address different issues:

YAML is used to store the configuration, i.e., the camera to be used along with its parameters,
Factory is used to select the corresponding implementation based on the name in YAML,
the Driver Interface is used as a common contract for all drivers,
and
the actual Driver is used to access the camera hardware.
Finally,
the Frame is produced as a result which is further consumed elsewhere in a standardized form.

YAML $ \rightarrow $ Factory $ \rightarrow $ Driver $ \rightarrow $ Standard Frame $ \rightarrow $ Application

This way, hardware-specific code is kept away from Capture Service, Tier1, Tier2, and scoring.

### How the Camera Interface was Designed

Recall that the initial goal was to support both laptop, raspberry pi, and jetson platforms in cameras without scattering the platform-specific code in all parts of the application. For this reason, we did some research to find out which drivers are available for each platform:

Laptop
├── Windows → Media Foundation
├── macOS → AVFoundation
└── Linux → V4L2

Raspberry Pi
├── USB → V4L2
└── CSI → libcamera / Picamera2

Jetson
├── USB → V4L2
└── CSI → Argus / GStreamer

As a result, we got that the choice of drivers depends on the camera technology, which is not surprising. Here are some examples of possible combinations:

Jetson + USB → webcam
Jetson + CSI → csi_gstreamer
Pi + USB → webcam
Pi + CSI → picamera

### Shared and Platform-specific Drivers

It turns out that for the case of USB + V4L2, this code can be shared between laptop, raspberry pi, and jetson. For the time being, we have decided to keep CSI-specific code separated for each platform:

webcam.py       → OpenCV / USB
rpi/picamera.py    → Picamera2 / Pi CSI
jetson/csi_gstreamer.py → Argus / GStreamer / Jetson CSI
mock/camera.py     → testing and development
This way, we avoid code duplication for the same technology (CSI) across different platforms (rpi, jetson) which would be necessary otherwise.

### Standard Frame Output

After reviewing the options for camera stacks, the next question was – what format should each driver return?

It is expected that different drivers return different structures:
```

OpenCV object
Picamera2 output
GStreamer buffer
```
In order not to tie downstream processing to a specific hardware stack, each driver converts its output into one standard `Frame` object:
```
Frame
├── data    → RGB numpy array
├── timestamp → capture time

├── source   → driver name
└── metadata  → additional info
```
The image data in the frame has the following format:
```
RGB
H × W × 3
uint8
```
This allows for a simple flow:
```
Different Camera Systems
↓
Different Drivers
↓
One Standard Frame
↓
Same AI Pipeline
```
### Failure Cases → Interface Design
The interface was designed based on the possible failure scenarios that can occur with cameras:
Camera opens, but does not output frames → verify real frames in `open()`
Bad first frames → warm up and skip first frames
Camera index changed → prefer `device_name`, fall back to `device_index`
Temporary read failures → retry on error
Camera dies later → `is_healthy()` checks for recent frames
Failure cases were identified, and based on them – interface requirements were established. Thus, an interface was implemented that encapsulated the requirements as code.
The process looked like this:
```
Research
↓
Failure cases
↓
Interface requirements
↓
Implementation
↓
Tests
```
This turned the research process into a living contract for camera drivers.
### Laptop Webcam Example
A developer can test a computer vision project on their own laptop:
They use the built-in or USB webcam;
```
Laptop Webcam
↓
OpenCV
↓
WebcamDriver
↓
CameraDriver
↓
Standard RGB Frame
↓
Capture Service
```
A minimal config for such a setup could look like this:
```
hardware:
driver:
camera: webcam
camera_params:
device_name: "Integrated Camera"
device_index: 0
width: 1280
height: 720
fps: 30
```

Since each laptop has its own camera name and index, such cameras must be discovered and verified using `detect_hardware.py` and `camera_probe.py`.

### Laptop Operational Note

To use a real webcam:
```

pip install pyyaml psutil opencv-python

python scripts/detect_hardware.py
python scripts/camera_probe.py --list
python scripts/camera_probe.py --driver webcam --frames 30
```

Once you successfully get 30 valid frames, the webcam driver and camera layer are working.

In case of no physical webcam, you could use:
```

camera: mock
```

This allows you to develop and test Capture → Tier1 → Tier2 → Risk Scoring chains using mocked frames.

### Raspberry Pi Operational Note

For USB webcam:
```

camera: webcam
```

The path is as following:
```

USB Camera → V4L2 → OpenCV → WebcamDriver → CameraDriver
```
You could check it by:
```

python3 scripts/detect_hardware.py
python3 scripts/camera_probe.py --list
python3 scripts/camera_probe.py --driver webcam --frames 30
```

For CSI camera:

Install Picamera2 first:
```

sudo apt install -y python3-picamera2
```

Use the config:
```

camera: picamera
```

The path is as following:
```

CSI Camera → libcamera → Picamera2 → PiCameraDriver → CameraDriver
```
You could check it by:
```

python3 scripts/camera_probe.py --list
python3 scripts/camera_probe.py --driver picamera --frames 30
```

### Jetson Operational Note

For USB webcam:
```

camera: webcam
```

The path is as following:
```

USB Camera → V4L2 → OpenCV → WebcamDriver → CameraDriver
```

For CSI camera:
```

camera: csi_gstreamer
```

The path is as following:
```

CSI Camera → Argus → nvarguscamerasrc → GStreamer → JetsonCSIDriver → CameraDriver
```

First verify the Jetson camera pipeline:
```

gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! fakesink
```

Then run:
```

python3 scripts/camera_probe.py --list
python3 scripts/camera_probe.py --driver csi_gstreamer --frames 30
```

### Camera Paths
```

Laptop
Webcam → webcam

Raspberry Pi
USB → webcam
CSI → picamera

Jetson
USB → webcam
CSI → csi_gstreamer
```

All drivers generate the same CameraDriver output and thus the same input format for the Capture Service and AI pipeline.

### architecture till camera interface 

```

             CONFIGURATION
                  │
                  ▼
          hardware.driver.camera
                  │
                  ▼
              FACTORY
                  │
          ┌───────┼────────┐
          ▼       ▼        ▼
       webcam  picamera  gstreamer
          │       │        │
          └───────┼────────┘
                  ▼
            CameraDriver
                  │
                  ▼
               Frame
                  │
                  ▼
           Capture Service
                  │
                  ▼
             Tier1 / Tier2


```

