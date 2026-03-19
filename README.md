# 超電脳サイバネティクス — SCSC

**SCSC Active Inference Humanoid Control System**

> 脳が予測し、身体が動く。

自由エネルギー原理に基づく Active Inference フレームワークで、17軸ヒューマノイドロボット（Manoi PF01）を全身協調制御するプロジェクトです。

📥 **[プレゼン資料をダウンロード（.pptx）](docs/scsc-pr-phase1-7.pptx)**

---

## Phase 1 — 7 Progress Report

### 1. タイトル
![Slide 1 - Title](docs/slides/slide-01.jpg)

### 2. Project Vision
![Slide 2 - Project Vision](docs/slides/slide-02.jpg)

### 3. Platform
![Slide 3 - Platform](docs/slides/slide-03.jpg)

### 4. Development Timeline
![Slide 4 - Development Timeline](docs/slides/slide-04.jpg)

### 5. Phase 6 — 顔追従 Active Inference
![Slide 5 - Phase 6 Face Tracking](docs/slides/slide-05.jpg)

### 6. Phase 7 — 全身 17軸 Active Inference 制御
![Slide 6 - Phase 7 Full Body Control](docs/slides/slide-06.jpg)

### 7. System Architecture
![Slide 7 - System Architecture](docs/slides/slide-07.jpg)

### 8. By the Numbers
![Slide 8 - By the Numbers](docs/slides/slide-08.jpg)

### 9. Safety Architecture
![Slide 9 - Safety Architecture](docs/slides/slide-09.jpg)

### 10. EFE Planner — 2段階行動計画
![Slide 10 - EFE Planner](docs/slides/slide-10.jpg)

### 11. Status & Next Steps
![Slide 11 - Status and Next Steps](docs/slides/slide-11.jpg)

### 12. エンディング
![Slide 12 - Ending](docs/slides/slide-12.jpg)

---

## Tech Stack

| Layer | Hardware | Role |
|-------|----------|------|
| 認知層 | Jetson AGX Orin | CUDA AIF 推論エンジン（5000粒子 × 100Hz） |
| 駆動層 | Teensy 4.1 | ICS 3.5 サーボ制御（リアルタイム 100Hz） |
| 身体層 | Manoi PF01 | KRS-4024S HV × 17軸 ヒューマノイド |

## References

- Thomas Parr, Giovanni Pezzulo, Karl J. Friston (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- 乾敏郎 訳『能動的推論 — 心、脳、行動の自由エネルギー原理』ミネルヴァ書房

## License

TBD
