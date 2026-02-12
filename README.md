# CoDeX-Net

Accurate waveform classification in hybrid radar-communication environments remains challenging, particularly under low-SNR conditions where structured interference and noise severely distort time-frequency signatures. Existing lightweight CNN models often lack the capacity to capture axis-dependent patterns, whereas conventional vision backbones are computationally prohibitive for edge-level deployment and fail to exploit the physical structure of spectrogram data. To address these limitations, this work proposes CoDeX-Net, a compact dual-stream architecture composed of two complementary modules: (i) PAUG, which performs intra-resolution refinement through coordinate-aware spatial gating and channel-adaptive modulation, and (ii) CSDM, a cross-scale deformable mixer that aggregates coarse-resolution context via offset-guided sampling and soft branch routing. Together, these modules enhance both local discriminability and long-range spectral coherence while maintaining extremely low complexity. Extensive experiments on twelve radar and communication waveforms demonstrate that CoDeX-Net achieves 91.01% average accuracy, outperforming state-of-the-art CNN and lightweight radio frequency (RF) classifiers despite operating with only 51K parameters and 0.564 ms inference latency. The results confirm that task-aligned architectural design provides substantive benefits over repurposed vision models and enables practical deployment in real-time embedded RF systems.

![Architecture](https://github.com/DatChanThanh/CoDeX-Net/blob/7ed3414ee0c155d04ca328c3a2f1805bff48b5e2/architecture.png)

![Architecture](https://github.com/DatChanThanh/CoDeX-Net/blob/7ed3414ee0c155d04ca328c3a2f1805bff48b5e2/PAUG.png)

![Architecture](https://github.com/DatChanThanh/CoDeX-Net/blob/7ed3414ee0c155d04ca328c3a2f1805bff48b5e2/CSDM.png)

The dataset can be download on [Google Drive](https://drive.google.com/drive/u/1/folders/15TJjTUcQEKmzlx7vJDgb7TK7XPZJZ5hF) (please report if not available).

 If there is any error or need to be discussed, please email to [Thanh-Dat Tran](https://github.com/DatChanThanh) via [trandatt21@gmail.com](mailto:trandatt21@gmail.com).
