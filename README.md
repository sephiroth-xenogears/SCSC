# SCSC Phase 7: Full-Body 17-Axis Active Inference Control

**超電脳サイバネティクス制御** — 脳が予測し、身体が動く。

## Overview

Phase 7 extends the Phase 6 face-tracking Active Inference engine from 2-axis (pan/tilt) to full-body 17-axis coordinated control on a Manoi PF01 humanoid robot.

| Component | Platform | Role |
|-----------|----------|------|
| Cognitive Layer | Jetson AGX Orin | Vision, AIF inference, action planning |
| Drive Layer | Teensy 4.1 | Servo communication, DMA, safety watchdog |
| Body Layer | Manoi PF01 | 17× KRS-4024S HV servos, ICS 3.5 |

## Architecture

```
Camera → FaceDetect → ObservationBuilder → AIF Engine → ServoCommander → Teensy → Servos
  (M1)     (M2)           (M3)           (M4-M7)         (M8)         (T1-T5)
                                            ↑                            |
                                    SafetyMonitor (M10)                  |
                                            ↑          servo feedback    |
                                            ←────────────────────────────┘
```

## Build (Jetson)

```bash
cd jetson
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Run

```bash
# Loopback mode (no Teensy required)
sudo ./build/phase7_main --loopback --log-dir ./logs

# With real servos
sudo ./build/phase7_main --real-servo --uart /dev/ttyTHS1 --baudrate 1250000
```

## Build (Teensy)

```bash
cd teensy
pio run -e teensy41
pio run -t upload
```

## Document Reference

- `SCSC-P7-ARCH-001` — Software Architecture Design (this codebase)
- `SCSC-P6-RPT` — Phase 6A Technical Report
