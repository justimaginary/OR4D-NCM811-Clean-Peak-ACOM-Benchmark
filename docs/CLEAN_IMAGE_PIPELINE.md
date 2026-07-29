# Clean diffraction-image pipeline

## Status

This is an experimental Clean-only extension of Clean-Peak v3. It has been
implemented and run on all 17 `legacy_smoke` orientations. It has **not**
passed its acceptance criteria, and no 1,081-pattern headline result is
claimed.

Dynamical scattering is out of scope. The existing Clean-Peak v3 data and
results remain unchanged.

## Four comparison paths

| Path | Input to ACOM | Purpose |
| --- | --- | --- |
| A | v3 Gaussian analytic peaks | Frozen historical baseline |
| B | Physical image-matched analytic peaks | Forward-model upper bound |
| C | Clean-E expectation → detector | Image formation and disk-finding loss |
| D | Clean-C counts → detector | Finite-dose loss |

Both C and D are run with the local AutoDisk chain and py4DSTEM 0.14.18
`find_Bragg_disks`.

## Clean-E forward model

Reciprocal coordinates use cycles per ångström and contain no `2π`. For a
sample-to-crystal orientation matrix \(R\),

\[
\mathbf g_\mathrm{sample}=R^\mathsf{T}\mathbf g_\mathrm{crystal}.
\]

For reflection \(g\), the aperture coordinate is
\(\boldsymbol\kappa=\mathbf q_\perp-\mathbf g_\perp\). The scattered
first-Born amplitude is

\[
\Psi_g(\mathbf q)=
F_g A(\boldsymbol\kappa)e^{-i\chi(\boldsymbol\kappa)}
t\,\mathrm{sinc}(t\Delta k_z)
e^{i\pi t\Delta k_z}.
\]

All reflection amplitudes are accumulated before intensity is evaluated:

\[
I_\mathrm{scattered}(\mathbf q)
=\left|\sum_g \Psi_g(\mathbf q)\right|^2.
\]

The direct and scattered images are normalized separately. The canonical
benchmark observation is then

\[
P(\mathbf q)
=f_0P_\mathrm{direct}(\mathbf q)
+(1-f_0)P_\mathrm{scattered}(\mathbf q),
\qquad f_0=0.90.
\]

The direct-beam fraction is explicitly a benchmark parameter, not a
first-Born depletion prediction. The configured sweep is 0.80, 0.90, and
0.98. The canonical detector is ideal: no PSF, background, read noise,
ellipticity, saturation, or gain variation.

At 300 kV and 0.5 mrad, the canonical 512×512 detector spanning
\(\pm1.6\,\text{Å}^{-1}\) has a diffraction-disk radius of approximately
4.06 pixels.

This model is more physical than drawing Gaussian points, but it is still an
ideal coherent first-Born model. With non-overlapping disks, zero aberration,
and no detector effects, its images will remain much cleaner and more regular
than experimental CBED.

## Clean-C observation model

The Clean-E image is normalized to a probability distribution. For a fixed
total dose \(N_e\),

```python
counts = rng.multinomial(N_e, expectation.ravel()).reshape(H, W)
```

Each image therefore satisfies `counts.sum() == N_e` exactly. The formal
doses are \(10^4\), \(10^5\), and \(10^6\) electrons per pattern, with five
deterministic independent repeats per dose. Counts are stored as `uint32`.
No individual electron event table is stored.

## Disk detectors and common output

AutoDisk:

```text
sqrt(image)
→ ring cross-correlation
→ LoG candidates
→ modified-RGM subpixel refinement
→ disk integration
```

py4DSTEM:

```text
image + ifftshift(vacuum_probe)
→ find_Bragg_disks
→ multicorr subpixel refinement
→ disk integration
```

py4DSTEM names detector array axes `qx` and `qy`; they correspond to array row
and column, not directly to this benchmark's horizontal physical \(q_x\) and
upward \(q_y\). The adapter performs the explicit row/column mapping and
corrects the half-pixel difference between py4DSTEM's even-array FFT origin
`floor(N/2)` and this benchmark's center `(N-1)/2`.

Candidate-localization scores are not passed to ACOM. Both methods integrate
the original image with the same relative disk radius and emit:

```text
sample_id
qx [Å⁻¹]
qy [Å⁻¹]
integrated intensity, max-normalized per pattern
```

## HDF5 files

Clean-E:

```text
public/clean_images.h5
├── expectation/intensity      [N,H,W] float32, sum=1
├── sample_id                  [N]
└── detector/
    ├── qx_Ainv                [W]
    ├── qy_Ainv                [H]
    ├── vacuum_probe           [H,W]
    └── valid_mask             [H,W]
```

Clean-C:

```text
public/clean_counted_images.h5
├── images/counts              [N,D,S,H,W] uint32
├── expectation/intensity      [N,H,W] float32
├── sample_id                  [N]
├── dose_electrons             [D]
├── rng_seed                   [N,D,S]
└── detector/...
```

Large generated images, detector HDF5 intermediates, and per-realization
submissions are ignored by Git and regenerated locally. Aggregate JSON reports
remain inspectable under `reports/`.

## Commands

Complete smoke run:

```bash
conda run -n or4d-clean ./run_clean_image_acom.sh smoke
```

Individual stages:

```bash
conda run -n or4d-clean \
  python scripts/02b_generate_clean_images.py --role legacy_smoke

conda run -n or4d-clean \
  python scripts/02c_generate_clean_counted_images.py \
  --expectation-file public/clean_images_smoke.h5 \
  --output public/clean_counted_images_smoke.h5

conda run -n or4d-clean \
  python scripts/03_extract_clean_disks.py \
  --image-file public/clean_images_smoke.h5 \
  --track expectation
```

`scripts/07_run_acom_baseline.py` accepts `--peak-file`,
`--prediction-file`, and `--report-tag`/`--output-tag`; every path uses the
same frozen 2° ACOM parameters. `scripts/15_run_clean_counted_acom.py` runs all
dose/seed/detector peak files and then aggregates them.

## Measured 17-case smoke results

Peak recovery against the current physical oracle:

| Input | Detector | Precision | Recall | RMSE (px) | P95 (px) |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean-E | AutoDisk | 0.921 | 0.463 | 0.427 | 0.646 |
| Clean-E | find_Bragg_disks | 0.983 | 0.857 | 0.182 | 0.463 |
| \(10^4\) e⁻ | AutoDisk | 0.158±0.005 | 0.127±0.004 | 0.583±0.016 | 0.938±0.017 |
| \(10^4\) e⁻ | find_Bragg_disks | 0.444±0.012 | 0.169±0.007 | 0.510±0.010 | 0.927±0.021 |
| \(10^5\) e⁻ | AutoDisk | 0.346±0.009 | 0.270±0.002 | 0.535±0.010 | 0.884±0.016 |
| \(10^5\) e⁻ | find_Bragg_disks | 0.482±0.010 | 0.384±0.006 | 0.460±0.008 | 0.889±0.012 |
| \(10^6\) e⁻ | AutoDisk | 0.740±0.008 | 0.427±0.005 | 0.486±0.005 | 0.784±0.020 |
| \(10^6\) e⁻ | find_Bragg_disks | 0.755±0.005 | 0.689±0.005 | 0.396±0.004 | 0.831±0.008 |

The five-repeat standard deviations above are real independent multinomial
realizations.

The physical-oracle ACOM upper bound is currently not valid:

| Input | Acc@2° | Acc@5° | >5° |
| --- | ---: | ---: | ---: |
| Physical oracle | 35.3% | 70.6% | 5/17 |
| Clean-E AutoDisk | 58.8% | 88.2% | 2/17 |
| Clean-E find_Bragg_disks | 52.9% | 88.2% | 2/17 |

The detector paths occasionally outperform the supposed oracle because disk
selection and re-integration change the weights sent to ACOM. That is evidence
of a model-contract problem, not evidence that detection improves the true
upper bound.

## Blocking issues before the full run

1. Reconcile the finite-thickness physical intensity definition with the
   excitation-error and intensity kernel used by the py4DSTEM orientation
   plan.
2. Define the physical oracle from the final observable image consistently;
   component-level reflection thresholds and detector-integrated thresholds
   are not yet fully equivalent.
3. Tune detector thresholds on a dedicated calibration split instead of
   silently optimizing them on headline orientations.
4. Re-run the 17 cases until path B is a credible upper bound and C agrees
   with B. Only then run 1,081 patterns and interpret Clean-C dose curves.
