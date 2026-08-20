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

# Capture Service 

Get frames from the camera driver, store what the frame is and when it occurred, have some history of the last few frames, give new frames to Tier 1, and then get back the precise frame that Tier 1 saw something in


```

CAMERA DRIVER
     │
     │ gives RGB frames
     ▼
CAPTURE SERVICE
     │
     ├── gives each frame an ID
     ├── records timestamps
     ├── stores recent frames
     ├── selects frames for Tier 1
     └── retrieves frames by ID
             │
             ▼
           TIER 1
             │
             │ "animal + bbox"
             ▼
       request original frame
             │
             ▼
      CAPTURE SERVICE
             │
             ▼
      original frame + bbox
             │
             ▼
           TIER 2

```


The Capture Service lives between the camera driver and Tier 1; it provides information about frames. It does not do object detection, it does not resize images, and it does not know about the AI model.

The camera driver is responsible for all the camera-specific logic, providing the Capture Service with a canonical RGB uint8 frame. The driver handles decoding, color spaces, hardware timestamps, etc. The Capture has no knowledge about cameras: it does not care if it is a webcam, CSI camera, or an RTSP stream, or that it provides 15, 30, or 60 FPS.

The Capture adds a unique ID to each frame: it is a combination of the source ID and the sequence number known only to the Capture, for example, cam01-000184392. The sequence number belongs to the Capture: the driver’s frame index is reset when the camera is disconnected and is not available anymore. The Capture timestamp can be used to determine the frame order across reboots.

A frame can have three timestamps:

media_timestamp: indicates when the frame was captured by a sensor and can be absent if not provided by the driver

received_timestamp: indicates the moment when the frame was received by the Capture using a monotonic clock. This timestamp is useful to calculate latency.

wall_clock: represents the current UTC time and is used to persist events to a database, logs, or for communication with external systems.

A missing media timestamp should not be guessed using another timestamp.

The Capture keeps a rolling buffer with the most recent original frames. When a new frame comes, it is added to the buffer, and the old one is discarded. The rolling buffer is mainly needed so that when Tier 1 detects an object, Capture can provide the original frame to Tier 2.

The buffer size should not be able to hold more than a minute of 1080p original frames. Assuming 6.2MB per 1920×1080 RGB frame, 30 FPS, and 60 seconds per minute, we get about 11GB per minute. The buffer should have a configurable retention time and memory size: the memory size has higher priority. For example, the system may want to keep the last 30 seconds, but if the system only has enough memory for 3 seconds, this is the hard limit. The longer video history should be stored in an encoded form in a video archive accessible to other services.

The Capture also selects which frames to pass to Tier 1. Suppose the camera provides 30 FPS, but only 5 FPS are needed; in that case, the Capture will drop the remaining 25 FPS. This should be done in time-based frames, not in every N-th frame, to handle cases when the camera FPS changes.

The Capture does not resize the image; it passes the canonical frame to Tier 1. Tier 1 is aware that it needs to perform resizing to, say, 320×320, so that the model can process it. It allows Capture to be ignorant of the model and its requirements.

The Tier 1 gets a frame from Capture, does some object detection, and returns a detection result with the frame ID to Tier 2. It can return information such as animal, confidence of 0.94, a bounding box, and the frame ID, for example, cam01-000184392.

The frame ID is critical; it allows Tier 2 to get the original frame from the Capture buffer. The Tier 1 result is placed in the Tier 2 queue. The Tier 2 gets this result and requests the original frame using the frame ID from the Capture buffer. It then crops the original image using the bounding box provided by the Tier 1 and passes this cropped frame to the Tier 2.

The sequence is as follows: the camera provides a frame to the Capture, which assigns the frame an ID and a timestamp and saves the original frame to the buffer. Then, Tier 1 gets a frame from the Capture buffer, does the object detection, and returns the results with the frame ID to Tier 2. Tier 2 asks for the original frame using the frame ID, gets it from Capture, crops the detected object using the bounding box, and provides this cropped object to Tier 2.

The Tier1 queue should only hold a small number of frames, ideally 1. If Tier1 starts dropping frames, old frames in the queue are removed when new ones come in; only the most recent frame is kept. This way, Tier1 is always processing the latest frames without waiting for older ones. At the same time, the dropped frames should be tracked so that the system knows that something is wrong, and Tier1 is too slow.

If the camera is disconnected while working, the Capture should close the driver, wait for some time (backoff), and reconnect; the sequence numbers are retained, so the Capture knows which frames have already been processed. However, the buffer with original frames is not cleared, and the Tier1 queue is cleared because all the frames there are now outdated.

The original frames in the rolling buffer are write-once, so that the memory can be shared between other processes if possible. Other services, including Tier1, get pointers to the original image data, not the copies of it, because these frames are the source of truth. If a service, such as Tier2, needs to modify the image, it should make a copy of the cropped region before making any changes. This way, other services continue to see the original image data.

Overall, the system is straightforward; all logic is in the Capture. It remembers what every frame is, when it was received, and where it is. All the Tier1 knows is that it has to process a certain canonical frame; it does not know where it came from. Once it finds the object in the frame, it tells Tier2 about it by giving the frame ID. Using this ID, Tier2 gets the original frame from Capture, crops the object using the bounding box, and passes it to Tier2.


```
                  CAMERA
                    │
                    ▼
              Capture Service
                    │
        ┌───────────┼────────────┐
        │           │            │
        ▼           ▼            ▼
       ID       timestamp      buffer
        │
        ▼
     Tier 1
        │
        │ "animal + bbox + ID"
        ▼
     Capture
        │
        │ "give me that exact ID"
        ▼
   Original frame
        │
        │ crop using bbox
        ▼
     Tier 2

```

The Capture Service comprises four files. Each file is responsible for a particular aspect to avoid having a single service.py file with all the code.

frame.py

This file defines what a captured frame is. It stores the image data alongside the identity and time details.

It handles frame_id, media_timestamp, received_timestamp, wall_clock, and the frame metadata. It also makes sure the stored image is read-only to avoid unnecessary modifications.

Simple explanation:

frame.py – What is this frame about?

buffer.py

This file handles the buffer of recent frames. It stores the captured frames, discards the old ones when the memory or time is exceeded, and provides a way to access a frame given the frame_id.

It also manages the three cases – frame available, frame expired, and frame unavailable.

Simple explanation:

buffer.py – What recent frames do we have?

sampler.py

This file determines which frames are to be passed to Tier 1. The camera may be capturing at 30 FPS, but Tier 1 only needs 5 FPS.

The sampler uses a time-based approach to capture the required number of frames instead of relying on the FPS provided by the camera driver. It also manages the small buffer for the Tier 1 queue to ensure that the latest frame is prioritized when there is a lag in the camera capture rate.

Simple explanation:

sampler.py – What frame does Tier 1 need?

service.py

This file defines the Capture Service. It connects all the components, provides a way to receive frames from the camera driver, adds IDs and timestamps to the captured frames, manages the buffer, processes the frames to be passed to Tier 1, handles the frame retrieval requests, manages camera reconnection, and provides health details.

It oversees the frame.py , buffer.py , and sampler.py files.

Simple explanation:

service.py – Run the Capture Service.

## operational thing

confirm the actual camera 

```
python scripts/camera_probe.py --driver webcam --frames 30

```

now run capture service 

```

python -m edge.capture

python -m edge --config config/laptop.yaml

```

for each service it depend 

Laptop

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

pytest tests/unit/ -v

python -m edge --config config/laptop.yaml

```

rasberry pi 

```

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

pytest tests/unit/ -v

python -m edge --config config/raspberry_pi.yaml

```

jetson

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

pytest tests/unit/ -v

python -m edge --config config/jetson.yaml

```
you do not have to run this again again 
if you want to specfically changes the particular service you can go for thing and test the service


# TIER 1/TIER 2 

## Tier 1 Detection

Tier 1 is the first stage of detection. Its job is to quickly check incoming frames and identify anything that might be worth investigating further.

The main goal here is **speed and filtering**, not perfect identification.

The flow is:

```text
Frame
  ↓
Motion gate (optional)
  ↓
Tier 1 detector
  ↓
Candidate
```

A `Candidate` contains basic information such as the detected category, confidence, and bounding box.

Tier 1 uses a coarse category system instead of trying to identify every possible species. This keeps the first stage lightweight and allows Tier 2 to do the more detailed verification.

Bounding boxes are stored using normalized coordinates rather than raw pixels. This keeps detections independent of the resolution of the original image.

---

## Tier 2 Verification

Tier 2 takes the candidates produced by Tier 1 and performs a more detailed verification.

Instead of running expensive verification on every frame, only the regions identified by Tier 1 are passed forward.

```text
Tier 1 Candidate
      ↓
Crop the relevant region
      ↓
Tier 2 verifier
      ↓
Verification
```

The crop is created while the original frame is still available. This is important because the capture buffer may no longer contain that frame later.

Tier 1 and Tier 2 use separate interfaces because they perform different jobs:

* **Tier 1:** find something potentially interesting.
* **Tier 2:** verify what that something actually is.

---

## Risk Scoring

After detection and verification, the system calculates a risk score.

The risk scorer uses deterministic rules and weighted factors instead of another AI model.

This is intentional. The same inputs should always produce the same score, and the reason behind a score should be understandable.

A score can therefore be broken down into factors such as:

```text
Detection confidence
Object/category
Movement
Location/context
Other configured risk factors
```

The scorer also provides an explanation for the result instead of returning only a single unexplained number.

This makes the final alert easier to understand and debug.

---

## Detection Events

Once the detection has been processed, the result is converted into a `DetectionEvent`.

The event is the standard format used by the rest of the system.

It contains information such as:

* Event ID
* Timestamp
* Detection information
* Normalized bounding box
* Risk score
* Tier 2 status
* Schema version

The event schema is treated as a stable contract because events may be stored locally before they are synchronized elsewhere.

Schema versioning is therefore included from the beginning rather than added later.

---

## Pipeline

The pipeline connects all the stages together:

```text
Camera Driver
     ↓
Capture Service
     ↓
Tier 1
     ↓
Tier 2
     ↓
Risk Scorer
     ↓
DetectionEvent
     ↓
EventBus
     ↓
Storage / Alerts / Other Subscribers
```

Each stage has a specific responsibility and communicates with the next stage through a defined interface.

The pipeline also uses queues between processing stages.

The queue policies are chosen based on the cost of the work already performed. Dropping an unused camera frame is different from dropping work that Tier 1 has already spent time processing.

This is especially important when Tier 2 is slower than Tier 1.

---

## Video File Driver

The system also supports analysing recorded video files such as MP4s.

The video driver follows the same capture interface as the other camera drivers, so the rest of the pipeline does not need to know whether a frame came from a live camera or a video file.

The main difference is that a video file has a definite end.

```text
Video file
   ↓
Frames
   ↓
End of stream
   ↓
Clean shutdown
```

End-of-stream is treated as a normal completion condition, not as a camera failure.

For recorded video, frame sampling uses the video's media timestamp rather than the computer's wall clock. This prevents fast playback or processing from causing large parts of the video to be skipped.

Video processing also uses blocking backpressure where reproducibility is required, so running the same recording multiple times produces consistent results.

---

## Testing

The system is tested around failure cases as well as normal operation.

Important cases include:

* Camera opens but never produces a valid frame
* Camera disconnects during capture
* Capture buffer expires before Tier 2 uses the frame
* Tier 2 is slower than Tier 1
* End of a video file
* Repeated analysis of the same video
* False detections on an empty/background-only video

The goal of the tests is not just code coverage. Each test protects against a specific failure mode that could otherwise reappear during future changes.

At the current stage, the project has **159 passing tests**.


# Operational Notes: Running Canopy Watch on a Laptop with an MP4

This walkthrough takes you from a fresh checkout to a complete Canopy Watch run using an MP4 file. You do not need a GPU, camera, or model files to get the basic pipeline working. The default Tier 1 and Tier 2 heuristic backends run locally using NumPy.

Real models are optional and can be added later.

## 1. Prerequisites

Check that Python is installed:

```bash
python3 --version
```

Python 3.9 or newer is fine.

It is also worth updating pip:

```bash
pip install --upgrade pip
```

You do not need a GPU, camera, or downloaded model files for this walkthrough.

---

## 2. Get the Repository and Install Dependencies

From the repository directory:

```bash
cd canopy-watch
pip install numpy opencv-python jsonschema
```

OpenCV is required because it handles reading the MP4 and writing snapshot crops.

`jsonschema` is technically optional because the event validator has a structural fallback, but installing it gives you more useful validation errors.

---

## 3. Run the Tests First

Before touching any video, make sure the environment itself is working:

```bash
python -m pytest tests/ -q
```

You should get:

```text
159 passed
```

If this fails before you have changed anything, treat it as an environment or installation problem first. There is no point debugging your video when the test suite is already on fire.

---

## 4. Get an MP4 to Analyze

There are two straightforward options.

### Option A: Generate Test Footage

If you do not have footage available, generate the synthetic test clip:

```bash
python scripts/make_test_video.py --seconds 15
```

This creates:

```text
tests/fixtures/test_clip.mp4
```

The clip contains a bright subject moving across a dark background. The subject enters around 20% of the way through the video and leaves around 80%.

It is useful for verifying the complete pipeline before introducing real-world footage.

### Option B: Use Real Footage

For example:

```bash
cp ~/Downloads/trail_cam_incident.mov clips/incident.mp4
```

If OpenCV later reports that it cannot open the file, the codec is usually the problem rather than the file container.

You can re-encode it with FFmpeg:

```bash
ffmpeg -i ~/Downloads/trail_cam_incident.mov \
    -c:v libx264 \
    -pix_fmt yuv420p \
    clips/incident.mp4
```

---

## 5. First Run: Keep It Simple

Run the generated test clip:

```bash
python scripts/analyze_video.py tests/fixtures/test_clip.mp4
```

You should see output roughly like:

```text
test_clip.mp4: 225 frames @ 15.0 fps (15.0s), tier1=heuristic tier2=heuristic/local motion_gate=off

   3.20s  [MEDIUM  ]  51.2  unknown  conf=0.99  video-analysis-01-000000048  top=confidence
   3.40s  [MEDIUM  ]  53.7  unknown  conf=0.99  video-analysis-01-000000051  top=confidence
   ...

--- summary ---
wall clock         2.1s
frames captured    224
frames to tier1    75
candidates         46
events             46
risk bands         {'medium': 40, 'high': 6}
peak score         62.3
```

A few things are worth understanding here.

**`frames to tier1`** is the number of frames actually examined. The default sampling interval is based on the video's timeline, not how quickly your laptop happens to process the video.

**`candidates` and `events` matching** means nothing was lost between candidate detection and event generation.

If `events` is zero, that does not automatically mean the pipeline is broken. The heuristic detector relies on brightness and blob-like motion, so a flat or low-contrast clip may simply contain nothing it can detect.

---

## 6. Run It Properly and Save the Results

Once the basic run works, save the outputs you will want to inspect later:

```bash
mkdir -p out

python scripts/analyze_video.py tests/fixtures/test_clip.mp4 \
    --snapshots out/snapshots \
    --events out/events.jsonl \
    --tz 5.5
```

The `--tz 5.5` setting represents UTC+5:30, or IST.

This matters because the timezone is used by the `nocturnal` risk factor. If you care about the actual risk score, use the correct timezone rather than treating it as decorative configuration.

Now inspect the generated files:

```bash
ls out/snapshots/
```

And inspect the first event:

```bash
cat out/events.jsonl | head -1 | python -m json.tool
```

Each JSON line contains the complete `DetectionEvent`, including:

* category
* confidence
* bounding box
* risk-factor breakdown
* human-readable risk reasons
* `snapshot_ref` pointing to the corresponding crop

---

## 7. Inspect an Actual Detection

Open one of the generated snapshots.

On macOS:

```bash
open out/snapshots/video-analysis-01-000000048.jpg
```

On Linux:

```bash
xdg-open out/snapshots/video-analysis-01-000000048.jpg
```

This is probably the most useful sanity check in the entire workflow.

If the crop clearly contains the subject, the detection pipeline is behaving sensibly even without a real ML model.

If the crop is mostly empty background, gray padding, or random noise, the Tier 1 threshold may be too loose or the heuristic detector may be responding to irrelevant motion.

A pipeline producing technically valid JSON while detecting absolutely nothing useful is still broken in the way that matters to humans.

---

## 8. Run a Negative Control

You also need to know how the system behaves when there is definitely no subject.

Generate an empty clip:

```bash
python scripts/make_test_video.py \
    --empty \
    --out tests/fixtures/empty_clip.mp4
```

Then analyze it:

```bash
python scripts/analyze_video.py tests/fixtures/empty_clip.mp4 -q
```

This clip contains only the background.

Therefore:

```text
candidates > 0
```

is a false positive.

This is the number to watch when tuning thresholds. Increasing detections is meaningless if false positives increase at the same time.

---

## 9. Understand File-Specific Sampling and Backpressure

For file-based testing, two options are particularly important:

```bash
python scripts/analyze_video.py clip.mp4 \
    --sample-clock media \
    --backpressure block
```

These are already the CLI defaults, but it is worth understanding why.

### `--sample-clock media`

Sampling follows the video's own timeline rather than wall-clock time.

This matters when a file is processed faster than real time.

For example, suppose an 8-second video takes only 1.4 seconds to process. Sampling based on wall time could result in roughly seven samples instead of the roughly 40 samples expected from a 200 ms media-time interval.

Using the media clock makes file analysis independent of how quickly the laptop happens to process the video.

### `--backpressure block`

This prevents frames from being dropped while Tier 1 is catching up.

That gives you repeatable results when analyzing the same file multiple times.

Reproducibility is one of the main reasons to test against a file instead of a live camera. A live camera has enough sources of chaos already without adding your pipeline to the list.

---

## 10. Tune the Threshold

Once the basic pipeline is working, compare several thresholds:

```bash
for t in 0.3 0.45 0.6; do
  echo "--- threshold $t ---"
  python scripts/analyze_video.py clip.mp4 \
      --threshold $t \
      -q | grep -E "candidates|events"
done
```

Because file processing is deterministic, differences between these runs should primarily reflect the threshold change rather than timing differences.

When tuning, do not look only at how many detections you get. Compare the results against the negative-control clip from Step 8.

---

## 11. Try the Motion Gate

The motion gate is disabled by default on a laptop:

```bash
python scripts/analyze_video.py clip.mp4 --motion-gate -q
```

Compare this run with the same command without `--motion-gate`.

Pay particular attention to:

* `frames_examined`
* `candidates`

The motion gate is intended to avoid running inference on static frames.

The tradeoff is straightforward: if a real subject is stationary, the motion gate can suppress it.

That is expected behavior, not necessarily a bug.

---

## 12. Optional: Use a Real Model

Once the heuristic pipeline is working, you can replace Tier 1 with a real model.

First see what models are available:

```bash
python scripts/fetch_models.py --list
```

Fetch the Tier 1 model:

```bash
python scripts/fetch_models.py --only tier1
```

Install an interpreter:

```bash
pip install tensorflow
```

Alternatively, use `tflite-runtime` if that is what your environment supports.

Then run:

```bash
python scripts/analyze_video.py clip.mp4 \
    --tier1 tflite_ssd \
    --tier1-model models/ssd_mobilenet_v2_coco_int8.tflite
```

The important part is that the rest of the pipeline does not change.

You keep the same:

* CLI
* event schema
* risk scoring
* snapshot handling
* downstream processing

Only the Tier 1 driver changes.

That is the practical test of whether the driver-swap architecture is actually doing what it claims to do.

---

## 13. Troubleshooting

| Symptom                                 | Likely cause                                                      | Fix                                                                                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OpenCV could not open <file>`          | Unsupported codec                                                 | Re-encode with `ffmpeg -c:v libx264 -pix_fmt yuv420p`                                                                                                 |
| Script hangs and never prints a summary | Old build without the `CameraEndOfStream` fix                     | Check that `edge/drivers/interfaces/camera.py` re-raises `CameraEndOfStream` before the generic `except Exception` in both `_warmup` and `read_frame` |
| Two runs produce different detections   | `--backpressure drop_oldest` was explicitly enabled               | Remove it or use `--backpressure block`                                                                                                               |
| `candidates: 0` on real footage         | Threshold is too high, or there is genuinely no detectable motion | Lower `--threshold` and verify with `--motion-gate` disabled                                                                                          |
| `frames dropped` appears in the summary | Backpressure is dropping frames                                   | Use `--backpressure block`                                                                                                                            |

---

## 14. What "Done" Looks Like

A successful end-to-end run should leave you with:

```text
out/events.jsonl
out/snapshots/*.jpg
```

The event file should contain schema-valid detection events.

The snapshot directory should contain visual evidence for detections above the configured minimum risk band.

You should also have:

* a false-positive baseline from the empty clip
* a repeatable command line
* deterministic results when running the same MP4
* confidence that the pipeline works before introducing real models

At that point, you have tested the actual workflow rather than merely proving that Python can open a file.

The final goal is simple: someone else should be able to take the same MP4, run the same command, and get the same output.


# MAKE IT SURVIVE STUPID INTERNET 

Every other part of the pipeline ( camera drivers,tier1/tier2,riskscoring) assumes the device is alive and working and this part assumes the opposite the device is on pi in field with flaky wifi or jetson relying on Lora/4G that drops for hours and detection must keep happeining no matter what the network is doing and nothing detected should ever be lost because a connection was not there at the moment it happened 

two question kept coming up 

what happens to a detection the instant its produced ?
what happens to a detection if the internet never comes back? 

a detection happened at that time will written to disk 
whether it reach to server it is separate later retryable concern 

the thing is what happen on device is fact and what leave is policy 

used SQLITE as three options before settling here:- 

1) in memory queue - faster but pi losing power mid queue loses every unsynced detection not acceptabe a detection that vanished silently is worse then no detection system at all 
2) flat json - durable but no easy say that particular thing is still pending without scanning and re writing the whole file and no atomic mark this one synced without a lock strategy i'd end up half reinventing SQlite anyway 
3) now SQLITE embedded no server process ACID ships with pythin runs fine on pi sd and wal mode lets capture pipeline write a new event while the sync agent reads pending ones and without the two blocking eachother 

hence used SQlite 

what about job queue

job ques implies a worker consumes and remove work but outbox implies the row is the detection record sunc just filps a bit on it and that distinction matters here because the events table isnot disposable infrastructure its device local hisotry of everything it has even detected deleting a row the moment it syncs would throw away a record you might later for debugging audits or resync  after a schema change on backend so synced is a column not a delete 

```

                  CANOPY WATCH EDGE DEVICE
┌─────────────────────────────────────────────────┐
│                                                   │
│  Camera                                          │
│    ↓                                             │
│  Capture Service                                 │
│    ↓                                             │
│  Tier 1 Detection                                │
│    ↓                                             │
│  Tier 2 Verification                             │
│    ↓                                             │
│  Risk Scoring                                     │
│    ↓                                             │
│  Event.new(...)                                  │
│    ↓                                             │
│  ┌──────────────────────┐                        │
│  │       SQLite          │                       │
│  │                        │                       │
│  │ EVT1 → synced          │                       │
│  │ EVT2 → pending         │                       │
│  │ EVT3 → pending         │                       │
│  └──────────┬─────────────┘                       │
│             │                                     │
│             ↓                                     │
│        SyncAgent                                  │
│             │                                     │
└─────────────┼─────────────────────────────────────┘
              │
         Internet?
          /      \
        NO        YES
        │          │
        │          ↓
        │      HTTPS Client
        │          │
        │          ↓
        │       Backend
        │          │
        │          ↓
        │     mark_synced()
        │
        └── keep locally


```

## failure case 

same process used for camera driver interface find failure and then build the interface around it rather then discovering the failure in the field and patching around it later 

internet is down for hours : - events must accumuate bounded by disk not memory and the device must not fall over just because the queue is growing Sqlite is on disk no inmemory buffering of unsynced events 

device loses power / restart mid backlog 

the outbox has to be readbale by fresh process with no shared state from the one that crash pending_events() only looks at the table nothing in process so as a restart just means a new reader against the same file 

sync agent itself throws 
bad response bug whatever 
this ust not take detection down with it the agent runs in its own thread with top level try/except around each loop iteration a crash there is logged and the loop continues next interval 

If 3 out of 10 events in a batch fail, don't try to force through 4-10 on the assumption that it won't. Run_once() aborts on the first failure within each pass.
If the event was already successfully transmitted, the backend will receive it again (backend agent received it, but connection was lost before the ack got back, agent retried. This means we will resend events - use an eventid which acts as the idempotency key - as an inserted eventid (if the detectionevent already exists locally on the backend - if this were a local database it could be an INSERT OR IGNORE, otherwise the backend must dedup based on eventid, by a configuration setting).
What if multiple field device events show up? Many field devices reconnect with power going to their entire site: if each site syncs with a constant retry, they may collide. To prevent this, backoff should be randomized. (This means an exponential delay for each site sync and also randomization).
A Pi with a low backoff limit may run through its sequence faster than a Jetson whose parameters require it to wait for up to 24 hours or more. So max_seconds for each device should come from its profile as part of other per-device configs. Laptop environments that are just getting setup or developers working in the device should have brief backoff for fast setup, whereas far-off devices should have long backoff so they aren't banging a live connection after the link goes down for months.
This client will be an interchangeable interface (and the same thinking is here as was present in the camera factories). The mode of communication, by default, should be as abstract as possible; the means of sending data from a device out into the cloud should not clutter the eventing loop. The SyncAgent cares nothing about whether it's sending via a real camera feed or something like MQTT/cloud streaming for larger-scale management use-cases. At any level it just cares about call on SyncClient.send(event) -> raise SyncError, nothing more. As seen in how tests work, if you just mock SyncClient and assign a value of online=False to the attribute you can test out-of-bound events on a simulated Pi with an intermittent (and ultimately failed) signal exactly the same way you can when testing a genuine Pi with no cell service.

## files??

Division of responsibility for event and file: We can make file responsibility a similar sort of breakdown as seen when moving out the capture service: one file one question. This leads us to a similar breakdown:

event.py - what are we syncing out?

Db.py - where do we keep it locally?

Outbox.py - which events have we not successfully synced yet, and how many are there?
Client.py - by what method(s) does the data get out to the backend?
Backoff.py - how long do we wait when we fail to get a send out successfully?
Agent.py - keep doing this over and over until it's done.
This means, once more, that a bug in "when will we wait" will not also slip into "when did we send."
Tests A test of end-to-end synchronization has been written as tests/testofflinesync.py, but this should not be interpreted as an "integration" or a "unit test", as much as an proof of system capability under an end-to-end, three-phase scenario.
Network OFF. Log 5 detection events on. Assert the 5 events exist locally and attempt at sending to an offline client generates no sent events, as we expect 5 will be held locally.
Disconnect from client. Reconnect to local persisted (non-persisted in memory, instead from File) database; restart everything, from Outbox down. Verify 5 pending messages persist and are ready for transmission on next opportunity.
Network ON. Agent autonomously begins to send; when all 5 events transmitted, they should have reached our client. This confirms none were sent in duplicate, and a proper zero is reported for outbox.pending_count.

## to test it up 

```
Python tests/testofflinesync.py
```
--- phase 1: network OFF, detections arriving ---

events stored locally: 5

sync attempt while offline: {'synced': 0, 'failed': 1, 'pending': 5}

--- phase 2: restart edge service (new process, same db) ---

events survived restart: 5

--- phase 3: network ON, agent syncs automatically ---

pending after sync: 0

sent via client: 5

PASS: internet OFF -> detection -> SQLite -> internet ON -> automatic sync

## Areas to consider moving forward include

Dead-letter queue/handling: Our events right now, if there's an unrecoverable error on the server, they’ll keep retrying until the max-backoff is hit each cycle. There should be a distinction between 5xx errors (retryable) and 4xx (non-retryable), and after N retries on a 4xx a hard stop for that event.
 Log retention for sent items: Sent items are stored locally with little fanfare for months, which may be suitable now but will take up space eventually. In much the same fashion we would handle video archives, we'll need a retention strategy that purges old items once some are sufficiently long off the device.
 Exposure of pending count: Our health check config requires an explicit measure that the number of pending items is not exceeding some threshold and some means to expose our pending queue's count. Currently it only offers outbox.pending_count() which is accessible by direct invocation; it does not appear in the health dashboard yet.
 MQTT-specific implementation of SyncClient: As was alluded to, we'll be looking into implementing a full range of MQTT functionality for fleet-managed solutions which will be where many future events and interactions land.
 At-rest payload encryption: If, in the future,DetectionEventPayload payloads do include personally identifying or similarly sensitive information, consider whether en-route and at-rest on disk encryption will be required. This is not something that will be present in initial deployments where events are unlikely to contain PII but it will likely emerge over time.

 