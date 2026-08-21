# CANOPY_WATCHER

```

 ██████╗ █████╗ ███╗   ██╗ ██████╗ ██████╗ ██╗   ██╗              
██╔════╝██╔══██╗████╗  ██║██╔═══██╗██╔══██╗╚██╗ ██╔╝              
██║     ███████║██╔██╗ ██║██║   ██║██████╔╝ ╚████╔╝               
██║     ██╔══██║██║╚██╗██║██║   ██║██╔═══╝   ╚██╔╝                
╚██████╗██║  ██║██║ ╚████║╚██████╔╝██║        ██║                 
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝        ╚═╝                 
                                                                  
██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗███████╗██████╗        
██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔══██╗       
██║ █╗ ██║███████║   ██║   ██║     ███████║█████╗  ██████╔╝       
██║███╗██║██╔══██║   ██║   ██║     ██╔══██║██╔══╝  ██╔══██╗       
╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║███████╗██║  ██║       
 ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝

```

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-ScientificComputing-013243?logo=numpy&logoColor=white)
![PyYAML](https://img.shields.io/badge/PyYAML-Configuration-CC0000)
![psutil](https://img.shields.io/badge/psutil-SystemMonitoring-3776AB)
![pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?logo=pytest&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-LocalDatabase-003B57?logo=sqlite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-FF6F00?logo=tensorflow&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-EdgeAI-FF6F00?logo=tensorflow&logoColor=white)
![GStreamer](https://img.shields.io/badge/GStreamer-MediaPipeline-FF6600?logo=gstreamer&logoColor=white)
![MQTT](https://img.shields.io/badge/MQTT-Messaging-660066?logo=mqtt&logoColor=white)

</p>

Computer Aided Network for Observing and Protecting Yielding wildlife habitats through Wireless AI-based Tracking and Camera Health monitoring



# WHAT IS CANOPY WATCH?

Canopy Watch is an offline-first edge AI wildlife monitoring system built for environments where the assumptions of camera, compute, storage, and network connectivity may not hold.

The system is centered around the single practical design goal that:

The animals need to be monitored whether the internet is available or not.

Field device captures camera images runs lightweight first-stage detector, performs computationally more costly verification only when necessary, calculate an explainable risk score, stores the derived event locally, and syncs to backend when connectivity present.

Desired application architecture must run on the: Laptop for simulation and development Raspberry Pi for restricted field applications Jetson Nano / Jetson Orin for accelerated edge inference

The key architectural design decision is that the differences in platforms are managed in configuration, hardware discovery, factories, and driver interfaces rather than scattering whether this is a Raspberry Pi, whether it is a Jetson or whether it's Windows branches throughout the application.


```
                    CANOPY WATCH
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     Camera          Edge AI          Local Storage
        │                │                │
        ▼                ▼                ▼
   Frame Capture → Tier 1 → Tier 2 → Risk Score
                                      │
                                      ▼
                                Event Database
                                      │
                              ┌───────┴───────┐
                              │               │
                           Offline          Online
                              │               │
                         Keep Local      Sync Backend

```
## Why this architecture?

A wildlife monitoring system deployed in the field has very different constraints from a normal computer-vision demo.

A laptop may have:

- reliable power
- fast SSD storage
- stable Wi-Fi
- plenty of CPU/GPU capacity
- easy debugging

A Raspberry Pi may have:

- limited CPU resources
- SD-card storage
- unreliable Wi-Fi
- no hardware accelerator
- optional battery/UPS information

A Jetson may have:

- GPU/TensorRT acceleration
- CSI camera hardware
- thermal throttling
- intermittent cellular or long-range connectivity

Canopy Watch therefore separates **facts** from **policy**.

For example:

- Hardware detection can determine that a GPU exists.
- Configuration decides whether Tier 2 should actually run locally.
- Hardware detection can discover storage and memory.
- Configuration decides retention and batching policy.

This distinction keeps machine-specific facts out of application logic and keeps operational decisions explicit.

---

# Core Architecture

```text
                         CANOPY WATCH
                     EDGE AI PIPELINE
                              │
                              ▼
                         Camera Driver
                              │
                              ▼
                       Capture Service
                              │
                              ▼
                         Tier 1 Detector
                              │
                       candidate + bbox
                              │
                              ▼
                         Tier 2 Verifier
                              │
                              ▼
                         Risk Scoring
                              │
                              ▼
                        DetectionEvent
                              │
                              ▼
                         SQLite Store
                              │
                              ▼
                          Sync Agent
                              │
                         Internet?
                         /        \
                       NO          YES
                       │            │
                       │            ▼
                       │       HTTP / Client
                       │            │
                       │            ▼
                       │      FastAPI Ingest
                       │            │
                       │     Validation/Auth
                       │            │
                       │     Deduplication
                       │            │
                       │            ▼
                       │       PostgreSQL
                       │
                       └── keep locally
```

The system is deliberately split into independent stages. Each stage has a narrow responsibility and communicates through defined contracts.

---

# 1. Hardware-Agnostic Configuration

Canopy Watch supports multiple deployment environments without changing the application code.

The configuration system is layered:

```text
base.yaml
    ↓
profile.yaml
    ↓
hardware.generated.yaml
    ↓
environment variables / overrides
    ↓
final validated configuration
```

### Configuration layers

| Layer | Purpose |
|---|---|
| `base.yaml` | Shared safe defaults |
| Profile YAML | Human-defined device policy |
| `hardware.generated.yaml` | Machine-detected facts |
| Environment variables | Secrets and deployment-specific overrides |

Typical profile configurations include:

```text
laptop
rpi
jetson_nano
jetson_orin
```

The generated hardware file is machine-specific and should not be treated as a manually maintained configuration source.

### Why this matters

The system needs to vary:

- node identity
- camera driver
- hardware tier
- accelerator
- model format
- storage policy
- networking
- retry/backoff behaviour
- thermal/battery monitoring
- logging and debugging behaviour

without changing the application itself.

---

# 2. Camera Driver Architecture

Camera hardware is intentionally isolated behind a common interface.

```text
Configuration
      │
      ▼
 Camera Factory
      │
      ├── webcam
      ├── picamera
      ├── csi_gstreamer
      └── mock
      │
      ▼
 CameraDriver
      │
      ▼
 Standard Frame
      │
      ▼
 Capture Service
```

Supported paths include:

| Platform | Camera | Driver |
|---|---|---|
| Laptop | USB/Webcam | OpenCV |
| Raspberry Pi | USB | OpenCV / V4L2 |
| Raspberry Pi | CSI | Picamera2 |
| Jetson | USB | OpenCV / V4L2 |
| Jetson | CSI | GStreamer / Argus |
| Development | Mock | Mock driver |

Every driver produces the same canonical frame representation:

```text
RGB
H × W × 3
uint8
```

A standard frame carries:

- image data
- timestamp information
- source/driver information
- metadata

This means downstream services do not need to know whether a frame originated from a webcam, CSI camera, or another capture source.

### Camera failure handling

The driver contract is designed around real failure cases:

- camera opens but produces no valid frames
- invalid initial frames
- camera index changes
- temporary read failures
- camera disconnects during operation
- camera becomes unhealthy after startup

The interface was therefore designed from:

```text
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

rather than treating camera drivers as simple wrappers around `cv2.VideoCapture`.

---

# 3. Capture Service

The Capture Service sits between camera drivers and AI inference.

Its job is to answer:

> **What frame was captured, when did it happen, and can I retrieve that exact frame later?**

```text
Camera Driver
      │
      ▼
Capture Service
      │
      ├── assign frame ID
      ├── record timestamps
      ├── maintain rolling buffer
      ├── sample frames for Tier 1
      ├── retrieve frames by ID
      └── reconnect failed cameras
      │
      ▼
Tier 1
```

The Capture Service deliberately does **not**:

- perform object detection
- resize images for a particular model
- know which AI model is running
- depend on the physical camera implementation

### Frame identity

Frames receive an ID such as:

```text
cam01-000184392
```

The sequence belongs to the Capture Service rather than the camera driver because driver frame counters may reset after reconnection.

### Timestamps

A frame may carry three distinct timestamps:

- `media_timestamp` - timestamp supplied by the source when available
- `received_timestamp` - when Capture received the frame
- `wall_clock` - UTC time used for persistence and external systems

A missing media timestamp is not silently guessed from another timestamp.

### Rolling buffer

The Capture Service keeps recent original frames in memory.

This is important because Tier 1 may detect something on a frame and Tier 2 may later need the **exact original frame**.

The buffer is:

- time bounded
- memory bounded
- configurable
- source-of-truth for recent frames

Memory limits take priority over requested retention time.

### Sampling

If a camera produces 30 FPS but Tier 1 only needs 5 FPS, Capture performs time-based sampling rather than blindly selecting every Nth frame.

That matters when camera FPS changes.

### Backpressure

Tier 1 is intentionally treated as a latest-frame consumer.

Its queue is kept small so that a slow detector does not spend its time processing stale frames.

When frames are dropped because Tier 1 cannot keep up, those drops should be observable as operational information rather than silently disappearing.

---

# 4. Tier 1 Detection

Tier 1 is the **fast filtering stage**.

Its job is not to perfectly identify an animal.

Its job is:

> **Find anything potentially interesting quickly.**

```text
Frame
  ↓
Optional Motion Gate
  ↓
Tier 1 Detector
  ↓
Candidate
```

A candidate contains information such as:

- category
- confidence
- normalized bounding box
- frame ID

Bounding boxes use normalized coordinates so detections remain independent of original image resolution.

The architecture allows Tier 1 to use a lightweight backend such as a TFLite model or the local heuristic backend used for development.

---

# 5. Tier 2 Verification

Tier 2 performs more expensive verification.

Instead of running expensive inference on every frame:

```text
Tier 1
   │
   │ candidate + bbox + frame_id
   ▼
Capture Service
   │
   │ original frame
   ▼
Crop detected region
   │
   ▼
Tier 2 Verifier
   │
   ▼
Verification result
```

The separation is intentional:

**Tier 1**

> "Something interesting may be here."

**Tier 2**

> "Let's spend more compute deciding what it actually is."

The crop is created while the original frame is still available, preventing the verification stage from depending on a frame that may already have expired from the rolling buffer.

Tier 2 can therefore be:

- local
- deferred to another compute environment
- enabled only on capable hardware

without changing the upstream Capture or Tier 1 interfaces.

---

# 6. Explainable Risk Scoring

Detection alone does not determine whether an event deserves attention.

Canopy Watch calculates a deterministic risk score after detection and verification.

The scorer intentionally uses **rules and weighted factors rather than another opaque AI model**.

Example factors include:

```text
Detection confidence
Object / category
Movement
Location / context
Other configured risk factors
```

The output is not only a number.

It also contains an explanation of why the score was produced.

That makes the system easier to:

- inspect
- debug
- tune
- audit
- explain to an operator

A deterministic scorer also means identical inputs produce reproducible scores.

---

# 7. Detection Events

After processing, the result becomes a `DetectionEvent`.

This is the stable event contract used by the rest of the system.

A detection event contains information such as:

```text
event_id
timestamp
detection information
normalized bounding box
risk score
risk explanation
Tier 2 status
schema version
snapshot reference
```

Schema versioning is included from the beginning because events may remain on an edge device for a long time before synchronization.

The event is therefore treated as a durable contract rather than an internal temporary object.

---

# 8. Offline-First Storage

This is one of the most important design decisions in Canopy Watch.

The system assumes:

> **The network will fail.**

A detection must not disappear because Wi-Fi, cellular connectivity, or a remote server is unavailable.

The edge device therefore writes the event to **SQLite first**.

```text
Detection
    ↓
SQLite
    ↓
Pending
    ↓
Sync Agent
    ↓
Backend
    ↓
Mark synced
```

SQLite was selected because it provides:

- local persistence
- ACID semantics
- no database server
- Python compatibility
- suitability for constrained devices
- WAL support for concurrent reads/writes

### Why not an in-memory queue?

Power loss could destroy every unsynchronized detection.

That is unacceptable for a surveillance system.

### Why not flat JSON?

A flat file makes it harder to efficiently:

- identify pending events
- atomically update sync state
- query local history
- handle concurrent access

SQLite provides those primitives without requiring a separate database service.

---

# 9. Outbox Synchronization

The local database acts as the source of truth until synchronization succeeds.

A synced event is **not deleted**.

Instead:

```text
event_001 → synced
event_002 → pending
event_003 → pending
```

This distinction matters because local history may be needed later for:

- debugging
- auditing
- resynchronization
- backend schema changes

The synchronization layer is split by responsibility:

```text
event.py
    What are we syncing?

db.py
    Where do we keep it locally?

outbox.py
    Which events are pending?

client.py
    How do events leave the device?

backoff.py
    How long do we wait after failure?

agent.py
    How do we keep trying?
```

The design prevents synchronization logic from contaminating the detection pipeline.

---

# 10. Failure-Tolerant Synchronization

The Sync Agent is designed for:

- hours without internet
- process restarts
- device power loss
- transient server failures
- lost acknowledgements
- multiple field devices reconnecting simultaneously

### Network offline

Events continue to accumulate in SQLite.

No in-memory-only backlog is required.

### Device restart

The new process reads the same SQLite database.

No previous in-process state is required to recover pending events.

### Sync agent failure

The sync loop is isolated so an exception in synchronization does not bring down detection.

### Lost acknowledgement

An event may be transmitted successfully but the acknowledgement may be lost.

The event is therefore retried.

The `event_id` acts as the idempotency key so the backend can deduplicate repeated transmissions.

### Backoff

Retry timing is configurable per deployment.

A developer laptop can retry quickly.

A remote field device can use much longer backoff intervals.

Randomization is used to avoid many field devices reconnecting simultaneously and hammering the backend.

---

# 11. Backend Ingestion

When connectivity is available:

```text
SQLite
   ↓
Sync Agent
   ↓
HTTP
   ↓
FastAPI Ingest API
   ↓
Validation / Auth
   ↓
Deduplication
   ↓
PostgreSQL
```

The backend provides the centralized representation of events from multiple edge devices.

The documented backend data model includes concepts such as:

- stations
- sightings
- alerts

The edge device remains autonomous while disconnected.

The backend becomes useful for aggregation, visualization, and fleet-level operations once data arrives.

---

# 12. Operator Dashboard

The dashboard exists for one reason:

> **Turn "the pipeline works" into "a human can see what happened."**

The dashboard is intentionally read-only.

```text
PostgreSQL
    ↓
FastAPI / Dashboard API
    ↓
dashboard/
    ├── index.html
    ├── style.css
    └── app.js
```

It does not create a second database or second write path.

It reads the same data already produced by the ingestion system.

### Operator-focused behaviour

The dashboard is designed to surface:

- recent sightings
- risk levels
- stations
- alerts
- event details
- filtering by station/risk/time

Risk is shown both as:

- a visual indicator
- a numeric value

This avoids relying entirely on color to communicate severity.

### Visible failure

If the API is unavailable, the dashboard can fall back to generated sample data while explicitly indicating that the data is not live.

The intent is to avoid the worst dashboard failure mode:

> a blank screen that looks like "there are no events" when the real problem is "the backend is dead."

### Current limitations

The documented dashboard does not yet provide:

- actual SMS/GSM alert dispatch
- dashboard/API authentication
- complete CORS configuration for real backend access

These are deployment concerns rather than reasons to duplicate the event data model.

---

# 13. Video File Analysis

Canopy Watch can analyze recorded video files such as MP4s using the same capture abstraction used by live cameras.

```text
MP4
 ↓
Video Driver
 ↓
Frames
 ↓
Capture
 ↓
Tier 1
 ↓
Tier 2
 ↓
Risk
 ↓
DetectionEvent
```

The rest of the pipeline does not need to know whether frames came from:

- a live camera
- a webcam
- a Raspberry Pi CSI camera
- a Jetson CSI camera
- a recorded video

### End of stream

End-of-file is treated as a normal completion condition.

It is not treated as a camera failure.

### Media-time sampling

For recorded video, sampling follows the video's own timeline rather than the laptop's wall clock.

This makes analysis reproducible even if an 8-second video takes 1 second or 20 seconds to process.

### Deterministic backpressure

Blocking backpressure can be used for file analysis so repeated runs of the same recording produce consistent results.

---

# 14. Quick Start: Laptop + MP4

You can run the core pipeline without:

- a GPU
- a physical camera
- downloaded model files

The development path uses local heuristic Tier 1/Tier 2 backends.

## 1. Clone the repository

```bash
git clone <repository-url>
cd canopy-watch
```

## 2. Create an environment

```bash
python3 -m venv .venv
```

### Linux/macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install the project

```bash
pip install -e .
```

If you are only testing the MP4 path in a minimal environment:

```bash
pip install numpy opencv-python jsonschema
```

## 4. Run the test suite

```bash
python -m pytest tests/ -q
```

The documented baseline is:

```text
159 passed
```

The test suite covers both normal operation and failure cases.

---

# 15. Generate Test Video

If you do not have a video:

```bash
python scripts/make_test_video.py --seconds 15
```

This generates:

```text
tests/fixtures/test_clip.mp4
```

The synthetic clip contains a bright moving subject on a dark background and is intended for validating the complete pipeline.

---

# 16. Analyze the Video

Run:

```bash
python scripts/analyze_video.py tests/fixtures/test_clip.mp4
```

The output reports information such as:

```text
frames captured
frames sent to Tier 1
candidates
events
risk bands
peak score
```

The exact numbers depend on the current implementation and configuration.

The important property is that the same input can be analyzed repeatedly.

---

# 17. Save Events and Snapshots

Create an output directory:

```bash
mkdir -p out
```

Then:

```bash
python scripts/analyze_video.py tests/fixtures/test_clip.mp4 \
    --snapshots out/snapshots \
    --events out/events.jsonl \
    --tz 5.5
```

This produces:

```text
out/
├── events.jsonl
└── snapshots/
    ├── ...
    └── ...
```

The event JSON contains information such as:

- category
- confidence
- bounding box
- risk-factor breakdown
- human-readable risk reasons
- snapshot reference

---

# 18. Inspect an Event

```bash
head -1 out/events.jsonl | python -m json.tool
```

A useful sanity check is to inspect the corresponding snapshot.

The goal is not merely to prove that JSON was generated.

The important question is:

> **Does the saved crop actually contain the thing the system claims it detected?**

A technically valid event pointing at empty background is still a failed detection system.

---

# 19. Run a Negative Control

Generate a background-only clip:

```bash
python scripts/make_test_video.py \
    --empty \
    --out tests/fixtures/empty_clip.mp4
```

Analyze it:

```bash
python scripts/analyze_video.py tests/fixtures/empty_clip.mp4 -q
```

The negative control is useful for measuring false positives.

When tuning thresholds, do not optimize only for more detections.

A useful detector must also avoid hallucinating animals out of shadows, compression artifacts, leaves, and the general chaos of nature.

---

# 20. Sampling and Backpressure

For deterministic file analysis:

```bash
python scripts/analyze_video.py clip.mp4 \
    --sample-clock media \
    --backpressure block
```

### `--sample-clock media`

Sampling follows the video's timeline.

This prevents processing speed from changing which parts of the video are sampled.

### `--backpressure block`

Frames are not discarded simply because Tier 1 is temporarily behind.

This is useful when reproducibility matters more than real-time throughput.

---

# 21. Threshold Tuning

Compare multiple thresholds:

```bash
for t in 0.3 0.45 0.6; do
  echo "--- threshold $t ---"
  python scripts/analyze_video.py clip.mp4 \
      --threshold $t \
      -q | grep -E "candidates|events"
done
```

Do this against both:

- a positive clip
- the negative-control clip

A threshold that doubles detections while also doubling false positives has not magically improved the system.

---

# 22. Motion Gate

The optional motion gate can be enabled with:

```bash
python scripts/analyze_video.py clip.mp4 --motion-gate -q
```

The motion gate is intended to avoid running inference on static frames.

The trade-off is unavoidable:

> A stationary animal may be suppressed by a motion-based filter.

Therefore, compare:

```text
motion gate OFF
vs
motion gate ON
```

using both positive and negative footage.

---

# 23. Optional Real Models

The heuristic backends are useful for validating the system architecture before introducing model complexity.

When ready, inspect available models:

```bash
python scripts/fetch_models.py --list
```

Fetch the Tier 1 model:

```bash
python scripts/fetch_models.py --only tier1
```

For a TensorFlow/TFLite-based Tier 1 backend:

```bash
pip install tensorflow
```

or use `tflite-runtime` where appropriate.

Example:

```bash
python scripts/analyze_video.py clip.mp4 \
    --tier1 tflite_ssd \
    --tier1-model models/ssd_mobilenet_v2_coco_int8.tflite
```

The important architectural property is that changing the Tier 1 implementation should not require rewriting:

- Capture
- event schema
- risk scoring
- snapshot handling
- downstream synchronization

Only the detector backend changes.

---

# 24. Hardware Deployment

## Laptop

Typical development path:

```bash
python scripts/detect_hardware.py
python scripts/camera_probe.py --list
python scripts/camera_probe.py --driver webcam --frames 30
```

For a real webcam, verify that valid frames are actually being produced.

Without a physical camera, use:

```yaml
camera: mock
```

This allows the rest of the pipeline to be tested without camera hardware.

---

## Raspberry Pi

### USB camera

```text
USB Camera
    ↓
V4L2
    ↓
OpenCV
    ↓
WebcamDriver
    ↓
CameraDriver
```

Probe it with:

```bash
python3 scripts/detect_hardware.py
python3 scripts/camera_probe.py --list
python3 scripts/camera_probe.py --driver webcam --frames 30
```

### CSI camera

Install Picamera2:

```bash
sudo apt install -y python3-picamera2
```

Configure:

```yaml
camera: picamera
```

Pipeline:

```text
CSI Camera
    ↓
libcamera
    ↓
Picamera2
    ↓
PiCameraDriver
    ↓
CameraDriver
```

Probe:

```bash
python3 scripts/camera_probe.py --list
python3 scripts/camera_probe.py --driver picamera --frames 30
```

---

## Jetson Nano / Orin

The Jetson should be running JetPack so the hardware detection logic can identify the platform correctly.

For CSI cameras:

```text
CSI Camera
    ↓
Argus
    ↓
nvarguscamerasrc
    ↓
GStreamer
    ↓
JetsonCSIDriver
    ↓
CameraDriver
```

Verify the camera pipeline:

```bash
gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! fakesink
```

Then:

```bash
python3 scripts/camera_probe.py --list
python3 scripts/camera_probe.py --driver csi_gstreamer --frames 30
```

For USB cameras, use the shared webcam driver.

---

# 25. Configuration Validation

Before starting a deployment, validate the merged configuration:

```bash
python -m edge.config.loader --profile <your-profile>
```

The loader is designed to fail fast.

For example, a field node without a required `node_id` should refuse to start instead of silently continuing with invalid identity information.

Environment variables can override deployment-specific settings.

Examples:

```bash
export CW_NODE_ID=rpi-field-03
export CW_PROFILE=rpi
export CW_SYNC_TOKEN=<provisioned-token>
```

Nested overrides follow the documented pattern:

```text
CW__section__key=value
```

For example:

```bash
export CW__networking__sync__endpoint=mqtt://gateway.local:1883
```

Secrets are intentionally referenced through environment variables rather than stored directly in committed YAML files.

---

# 26. Repository Structure

The architecture follows the same principle used throughout the project:

> **One component, one responsibility.**

A representative structure is:

```text
canopy-watch/
│
├── config/
│   ├── base.yaml
│   ├── laptop.yaml
│   ├── rpi.yaml
│   ├── jetson_nano.yaml
│   ├── jetson_orin.yaml
│   └── hardware.generated.yaml
│
├── edge/
│   ├── config/
│   │   └── loader.py
│   │
│   ├── drivers/
│   │   ├── webcam.py
│   │   ├── video.py
│   │   ├── mock/
│   │   ├── rpi/
│   │   └── jetson/
│   │
│   ├── capture/
│   │   ├── frame.py
│   │   ├── buffer.py
│   │   ├── sampler.py
│   │   └── service.py
│   │
│   ├── inference/
│   │   ├── tier1/
│   │   └── tier2/
│   │
│   ├── events/
│   ├── storage/
│   ├── sync/
│   │   ├── event.py
│   │   ├── db.py
│   │   ├── outbox.py
│   │   ├── client.py
│   │   ├── backoff.py
│   │   └── agent.py
│   │
│   └── ...
│
├── backend/
│   ├── ingest/
│   └── dashboard.py
│
├── dashboard/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── scripts/
│   ├── detect_hardware.py
│   ├── camera_probe.py
│   ├── analyze_video.py
│   ├── make_test_video.py
│   └── fetch_models.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── ...
│
└── infra/
    └── ...
```

The exact repository tree should be kept aligned with the implementation as files are added or renamed.

---

# 27. Testing Philosophy

Canopy Watch does not treat testing as simply a line-coverage exercise.

Tests are built around failure modes.

Important cases include:

- camera opens but never produces a valid frame
- camera disconnects during capture
- expired frame requested by Tier 2
- Tier 2 slower than Tier 1
- video reaches end-of-stream
- repeated analysis of the same recording
- empty/background-only footage
- network unavailable
- persisted events survive restart
- synchronization resumes after reconnection
- duplicate transmissions remain idempotent

The documented project state includes:

```text
159 passing tests
```

The goal is to protect system behaviour, not merely make the test counter look impressive.

---

# 28. End-to-End Offline Sync Test

The synchronization test validates three phases.

### Phase 1: Network OFF

```text
5 detections
    ↓
SQLite
    ↓
5 pending events
```

Expected:

```text
events stored locally: 5
pending: 5
synced: 0
```

### Phase 2: Restart

Start a fresh process using the same database.

Expected:

```text
events survived restart: 5
```

This proves that pending state is persisted rather than held only in memory.

### Phase 3: Network ON

The Sync Agent resumes transmission.

Expected:

```text
pending after sync: 0
sent via client: 5
```

The documented result is:

```text
PASS: internet OFF -> detection -> SQLite -> internet ON -> automatic sync
```

---

# 29. Operational Principles

Canopy Watch is built around a few non-negotiable principles.

### 1. Detection must not depend on the network

The camera and inference pipeline continue operating while the network is unavailable.

### 2. Local persistence comes before synchronization

A detection becomes durable locally before the system worries about sending it elsewhere.

### 3. Hardware-specific logic stays at the edge

Camera stacks, accelerators, and platform details should not leak into generic application code.

### 4. Interfaces are designed around failure

Camera drivers and synchronization clients are contracts built from real failure cases.

### 5. Tier 1 optimizes for throughput

Fast filtering prevents expensive inference from being wasted on every frame.

### 6. Tier 2 optimizes for verification

Expensive compute is reserved for candidates.

### 7. Risk scoring remains explainable

The final score should have understandable contributing factors.

### 8. Events are durable contracts

Schema versioning allows events to survive synchronization delays and backend evolution.

### 9. Synchronization is retryable, not destructive

Synced records remain available locally rather than disappearing immediately after transmission.

### 10. The dashboard is a view, not another source of truth

It reads the existing event data rather than creating a competing data store.

---

# 30. Current Limitations and Future Work

The project documentation identifies several areas that still need work.

### Configuration

- dependency pinning should be kept current
- Windows/macOS camera enumeration needs broader automation
- fleet-level configuration push is not part of the single-node loader yet

### Synchronization

- distinguish retryable `5xx` failures from non-retryable `4xx` failures
- introduce dead-letter handling after repeated permanent failures
- add retention policies for old synced events
- expose pending event count in health monitoring
- implement a fuller MQTT-based synchronization client
- consider encryption for sensitive payloads if future event data contains PII

### Dashboard

- add authentication
- configure CORS for production backend access
- connect real alert dispatch mechanisms
- continue aligning dashboard SQL with the current backend schema

These are deliberately separated from the core edge detection path.

---

# 31. What "Done" Means

A successful local end-to-end demonstration should produce:

```text
out/
├── events.jsonl
└── snapshots/
    └── *.jpg
```

and provide:

- schema-valid detection events
- visual evidence for relevant detections
- a false-positive baseline
- deterministic MP4 analysis
- working Capture → Tier 1 → Tier 2 → Risk pipeline
- persistent local event storage
- recoverable synchronization
- a readable operator view

The strongest demonstration is not:

> "The script ran without crashing."

It is:

```text
Camera / MP4
     ↓
Capture
     ↓
Detection
     ↓
Verification
     ↓
Risk
     ↓
Event
     ↓
Local persistence
     ↓
Network failure
     ↓
Restart
     ↓
Network recovery
     ↓
Backend
     ↓
Dashboard
```

with the original detection still present and traceable at the end.

---

# 32. Development Checklist

```text
[ ] Install project dependencies
[ ] Run the complete test suite
[ ] Validate hardware configuration
[ ] Probe the camera
[ ] Run Capture Service
[ ] Run Tier 1
[ ] Run Tier 2
[ ] Verify risk scoring
[ ] Generate DetectionEvent
[ ] Persist event locally
[ ] Simulate network failure
[ ] Restart process
[ ] Confirm pending events survived
[ ] Restore network
[ ] Confirm automatic synchronization
[ ] Verify backend record
[ ] Open dashboard
[ ] Inspect the detection snapshot
```

---


# Future Roadmap

After the **MVP pipeline is working**, Canopy Watch moves from:

> **Prove the architecture → make it production-like.**

The MVP should first prove the complete path:

```text
One camera
   ↓
One detection
   ↓
One event
   ↓
Local persistence
   ↓
Synchronization
   ↓
Backend
   ↓
Dashboard
```

Only after that path is reliable should the system expand into advanced intelligence, event streaming, ML lifecycle management, hardware optimization, and fleet-scale infrastructure.

---

## Phase 5 — Make the Edge Actually Intelligent

The first post-MVP step is improving the intelligence running directly on the edge device.

### Goals

- Upgrade the Tier 1 model and benchmark FPS/latency.
- Implement the real Tier 2 verification model.
- Trigger Tier 2 only when Tier 1 finds a candidate.
- Add proper bounding boxes, classes, and confidence values.
- Improve Risk Scoring using:
  - detection confidence
  - species/class
  - location/zone
  - time of day
  - previous sightings
  - repeated detections
- Add temporal tracking so the same animal is not counted repeatedly.
- Add configurable risk thresholds.

### Target pipeline

```text
Camera
 ↓
Tier 1
 ↓ candidate?
 ├── NO → discard
 └── YES
      ↓
    Tier 2
      ↓
   Tracking
      ↓
  Risk Score
      ↓
    Event
```

**Milestone:** Canopy Watch moves beyond basic detection into a genuine wildlife-intelligence pipeline, where detections have temporal context and repeated observations can be interpreted as one animal or one activity sequence rather than unrelated events.

---

## Phase 6 — Real Event-Driven Cloud

The MVP can begin with REST and simple consumers. The next step is introducing the event-driven infrastructure described by the architecture.

### Planned infrastructure

- Introduce **Kafka**.
- Create dedicated topics:
  - `edge.detections`
  - `edge.health`
  - `edge.heartbeat`
  - `enriched.detections`
  - `alerts`
- Add consumer groups.
- Add partitioning by camera/zone.
- Add event replay.
- Add schema compatibility and versioning.
- Replace the simple stream consumer with **Flink / Kafka Streams**.
- Implement event-time processing.
- Correlate sightings across cameras.
- Detect repeated sightings and movement patterns.

### Target architecture

```text
Camera A ─┐
          ├──→ Kafka ──→ Stream Processor ──→ Enriched Events
Camera B ─┘                  │
                             ├──→ Correlation
                             ├──→ Aggregation
                             └──→ Alerts
```

**Milestone:** The system can reason across **multiple cameras and streams**, rather than processing every detection as an isolated event.

---

## Phase 7 — Real Intelligence + Explainability

Once detection and event processing are reliable, the next step is richer contextual reasoning.

### Planned capabilities

- Sophisticated Risk Scoring.
- Contextual features.
- Historical behaviour.
- Zone-based risk.
- Anomaly detection.
- SHAP/LIME-based explainability.
- Store risk explanations alongside events.
- Show operators **why** an event was considered high-risk.

Example:

```text
Risk Score: 87/100

Reasons:
├── High-confidence animal detection
├── Restricted zone
├── Nighttime
├── Repeated sighting
└── Previous activity in this area
```

The architecture includes an Explainability Service for feature attribution.

**Milestone:** The system does not merely report **“High risk.”** It can explain why the event received that score.

---

## Phase 8 — Production Data Pipeline

Once the system has accumulated meaningful detections, the data itself becomes a first-class asset.

```text
Raw Events
    ↓
Bronze
    ↓
Silver
    ↓
Gold
    ↓
Training Dataset
```

### Planned work

- Create S3/object storage.
- Implement Bronze dataset.
- Implement Silver cleaning and normalization.
- Implement Gold training dataset.
- Store detection images/crops.
- Store metadata alongside images.
- Build ETL jobs.
- Version datasets.
- Track model performance.

The goal is to transform operational detections into reproducible datasets that can be used for model improvement.

---

## Phase 9 — Retraining + Model Registry

This phase closes the ML feedback loop.

```text
Edge
 ↓
Detections
 ↓
Gold Dataset
 ↓
Retraining
 ↓
Model Evaluation
 ↓
Model Registry
 ↓
New Model
 ↓
Edge OTA
```

### Planned capabilities

- Train new models from Gold data.
- Evaluate new models against the previous production model.
- Track evaluation metrics.
- Version model artifacts.
- Create a Model Registry.
- Store:
  - model version
  - dataset version
  - metrics
  - training configuration
- Select the production model.
- Implement model download on edge devices.

**Milestone:** The system can improve its models from deployment data while retaining traceability between the model, dataset, and evaluation results.

---

## Phase 10 — Jetson Deployment

Jetson deployment should happen **after the laptop MVP and cloud pipeline are working**.

### Planned work

- Implement the Jetson CSI camera driver.
- Implement GPIO driver support.
- Implement power monitoring.
- Create `nano.yaml`.
- Create `orin.yaml`.
- Optimize inference.
- Convert models to TensorRT or the appropriate Jetson runtime.
- Benchmark:
  - FPS
  - latency
  - RAM
  - GPU utilization
  - power consumption
  - temperature
- Package services using systemd and/or containers.
- Run completely offline.
- Synchronize automatically when connectivity returns.

### Target deployment

```text
Laptop Version
      ↓
Same Application
      ↓
Jetson Nano / Orin
      ↓
CSI Camera
      ↓
Offline Operation
      ↓
Automatic Sync
```

**Milestone:** The application runs on Jetson hardware using the same core interfaces and event contracts established during development.

---

## Phase 11 — Production Infrastructure

Only after the application is reliable should the infrastructure become substantially more complex.

### Planned infrastructure

- Dockerize services properly.
- Kubernetes deployment.
- Production PostgreSQL configuration.
- Kafka cluster.
- S3/object storage.
- Redis where it provides a real benefit.
- Secrets management.
- TLS.
- Device authentication.
- Monitoring.
- Centralized logging.
- Metrics.
- Health checks.
- CI/CD.
- Database migrations.
- Backup and recovery.

Infrastructure should solve an actual scaling or operational problem. Adding Kubernetes to a single laptop because the YAML looks impressive is not architecture. It is decorative suffering.

---

## Phase 12 — Fleet Management

Once one Jetson node works reliably, the problem changes from:

> **Can one device run?**

to:

> **Can many devices be operated safely?**

### Planned capabilities

- Device registration.
- Device certificates.
- Device heartbeat.
- Remote health status.
- Remote configuration.
- Model rollout.
- OTA updates.
- Rollback.
- Canary deployment.
- Device groups.
- Remote diagnostics.

### Target lifecycle

```text
Device Registration
       ↓
Configuration
       ↓
Deployment
       ↓
Heartbeat
       ↓
Health Monitoring
       ↓
Model / Software Update
       ↓
Canary Rollout
       ↓
Full Rollout
       ↓
Rollback if required
```

The architecture eventually calls for dedicated edge-fleet management and canary OTA rollout.

---

## Phase 13 — Advanced Scalability

This is the genuinely later-stage work.

### Possible directions

- Multi-park support.
- Multi-tenant architecture.
- `org_id` partitioning.
- Multiple ranger teams.
- Geographic zones.
- Cross-camera tracking.
- Federated learning.
- Alternative edge accelerators.
- Hailo support.
- Coral support.
- Large edge-device fleets.

These capabilities should come after the single-node architecture, event model, synchronization layer, and deployment workflow are stable.

---

# Complete Roadmap

```text
PHASE 0
Skeleton
   ↓
PHASE 1
Edge Detection
   ↓
PHASE 2
Offline-First Storage + Sync
   ↓
PHASE 3
Cloud Backend
   ↓
PHASE 4
Dashboard + Alerts
   ↓
════════════ MVP COMPLETE ════════════
   ↓
PHASE 5
Better Tier 1 + Real Tier 2 + Tracking
   ↓
PHASE 6
Kafka + Stream Processing
   ↓
PHASE 7
Advanced Risk + Explainability
   ↓
PHASE 8
Bronze → Silver → Gold Data Lake
   ↓
PHASE 9
Retraining + Model Registry + OTA
   ↓
PHASE 10
Jetson Deployment + Optimization
   ↓
PHASE 11
Kubernetes + Production Infrastructure
   ↓
PHASE 12
Fleet Management
   ↓
PHASE 13
Multi-Park + Federated Learning + Advanced Hardware
```

## The 80/20 Progression

| Stage | Primary goal |
|---|---|
| **First 20%** | Make one camera → one detection → one event → one sync → one dashboard entry work |
| **Next 30%** | Make the edge intelligent, explainable, and reliable |
| **Next 30%** | Make the system scalable, data-driven, and deployable |
| **Final 20%** | Make it research/production-grade across fleets and environments |

The order matters.

The architecture is a **blueprint**, not a requirement to implement every box immediately. The MVP proves that the core contracts work. Later phases increase intelligence, scale, operational maturity, and deployment reach without requiring the foundation to be rewritten.

---

# 33. Design Summary

Canopy Watch is not primarily a camera application.

It is an **edge reliability system with computer vision inside it**.

The important architecture is:

```text
                  HARDWARE
                     │
                     ▼
              Camera Drivers
                     │
                     ▼
             Capture Service
                     │
                     ▼
               Tier 1 AI
                     │
                     ▼
               Tier 2 AI
                     │
                     ▼
              Risk Scoring
                     │
                     ▼
            Detection Events
                     │
                     ▼
              SQLite Outbox
                     │
              ┌──────┴──────┐
              │             │
           Offline         Online
              │             │
              │             ▼
              │         Sync Agent
              │             │
              │             ▼
              │          FastAPI
              │             │
              │             ▼
              │        PostgreSQL
              │             │
              │             ▼
              └──────► Dashboard
```

The system is designed so that the most important fact in the system,

> **"this device detected something at this time"**

does not depend on whether the internet happened to be working at that exact moment.

That is the central engineering decision behind Canopy Watch.

---
