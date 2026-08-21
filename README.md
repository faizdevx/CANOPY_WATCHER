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
