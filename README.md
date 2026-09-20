# uncertainty-aware-regression

Core implementation of the deep-learning betatron-tune estimator described in

> P. Sun, M. Zhang, R. Yuan, D. Li, J. Dong and Y. Shi, *Real-time-capable betatron tune measurement from Schottky spectra using deep learning and uncertainty-aware Kalman filtering*, JINST **21** (2026) P02049, [doi:10.1088/1748-0221/21/02/P02049](https://doi.org/10.1088/1748-0221/21/02/P02049)

(sections "Neural network architecture" and "Training and uncertainty-aware filtering"). JINST **21** (2026) P08005 refers to it as CNN+KF.

## Method

The uncertainty-aware regression (UAR) estimator reads the betatron tune from Schottky spectra folded onto the tune axis q ∈ [0, 0.5).

1. **Input.** Two channels of length L: the robustly normalised PSD and the soft-binning weight map.
2. **Network.** A compact 1-D CNN with parallel fine and coarse stems, a shared residual body and two branches. The tune branch forms attention weights over the tune grid and returns their weighted mean, which is a differentiable soft-argmax. The uncertainty branch combines global difficulty statistics with the attention entropy and returns a per-frame σ.
3. **Training objective.** A Laplace negative log-likelihood, so that σ is calibrated against the actual error.
4. **Temporal filter.** A scalar Kalman filter with a random-walk model takes σ² of each frame as its measurement noise, so uncertain frames receive a small gain.
5. **Inference engine.** The forward pass is captured in a CUDA Graph and replayed for every frame.

## Files

| File | Content |
|---|---|
| `models.py` | `TuneMeasurementCNN`: stems, shared body, tune branch, uncertainty branch |
| `config.py` | Architecture and training hyperparameters |
| `loss.py` | `LaplaceNLL` |
| `uar_helpers.py` | `KalmanTuneTracker` and the cached input feature builder |
| `uar_estimator.py` | `UAREstimator` (`process(frame) -> MeasurementResult`): input construction, inference, Kalman filtering |
| `inference.py` | `InferenceEngine`, the CUDA Graph replay path |
| `tune_pipeline/cnn_features.py` | `FeatureBuilder`: MAD z-score channel and normalised weight channel |
| `tune_pipeline/mapping/mad.py` | MAD normalisation over the valid tune region |
| `tune_pipeline/frames.py`, `tune_pipeline/conventions.py` | Data contracts and the low-tune cutoff |

The modules import each other by file name, so the repository root has to be on `sys.path`.

## Scope

This repository contains the network, the loss, the Kalman filter and the inference path. The spectral preprocessing that produces a `FrontendFrame` (paper section "Data preprocessing pipeline"), the training loop, the datasets and the trained weights are not included.

## Dependencies

PyTorch, NumPy, Numba.

## Citation

```bibtex
@article{sun2026realtimecapable,
  title     = {Real-time-capable betatron tune measurement from Schottky spectra using deep learning and uncertainty-aware Kalman filtering},
  author    = {Sun, Peihan and Zhang, Manzhou and Yuan, Renxian and Li, Deming and Dong, Jian and Shi, Ying},
  journal   = {Journal of Instrumentation},
  volume    = {21},
  number    = {02},
  pages     = {P02049},
  year      = {2026},
  publisher = {IOP Publishing}
}
```

## License

MIT, see [LICENSE](LICENSE).
