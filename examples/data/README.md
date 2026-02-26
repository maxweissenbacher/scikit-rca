Synthetic example dataset for `examples/run_rca.py`.

Files:
- `features.npy`: shape `(288, 496)`, `float32`
- `labels.npy`: shape `(288, 2)`, `int64`

Dataset layout:
- 72 synthetic subjects
- 4 repeated scans per subject
- 496 features per scan (upper-triangle edges from a 32-node connectome: `32 * 31 / 2`)

This dataset is synthetic and was generated to be broadly similar in form to resting-state functional connectivity inputs used for RCA:
- repeated scans per subject
- connectome-edge-style feature vectors
- subtle repeat-consistent subject signal mixed with larger nuisance/state variation

It is not derived from the paper's real data.
