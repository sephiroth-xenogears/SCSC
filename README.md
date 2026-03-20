# SCSC — 超電脳サイバネティクス制御システム
**Super Cybernetic System Controller — CUDA-Powered Active Inference**

> 能動的推論 (Active Inference) を NVIDIA CUDA で実装し、二足歩行ヒューマノイドロボットをリアルタイム自律制御するシステム

## Platform
`NVIDIA CUDA` `Jetson AGX Orin` `Teensy 4.1` `Manoi PF01` `Active Inference`

---

## Portfolio

![slide-01](docs/slides/slide-01.jpg)
![slide-02](docs/slides/slide-02.jpg)
![slide-03](docs/slides/slide-03.jpg)
![slide-04](docs/slides/slide-04.jpg)
![slide-05](docs/slides/slide-05.jpg)
![slide-06](docs/slides/slide-06.jpg)
![slide-07](docs/slides/slide-07.jpg)
![slide-08](docs/slides/slide-08.jpg)
![slide-09](docs/slides/slide-09.jpg)
![slide-10](docs/slides/slide-10.jpg)
![slide-11](docs/slides/slide-11.jpg)
![slide-12](docs/slides/slide-12.jpg)

---

## Repository Structure

```
SCSC/
├── README.md
├── src/                          # Phase 7 ソースコード
│   ├── gen_model.cuh             # CUDA 生成モデル
│   ├── ics_driver.cpp            # ICS サーボ通信
│   └── main.cpp                  # メインループ
├── docs/
│   ├── scsc-pr-phase1-7.pptx     # PR プレゼンテーション
│   ├── scsc-portfolio.pptx       # ポートフォリオ
│   └── slides/                   # README 埋め込み用画像
└── .gitignore
```

## License
This project is for personal/educational purposes.
