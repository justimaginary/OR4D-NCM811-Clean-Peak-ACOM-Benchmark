# Clean diffraction-image pipeline

## Status

This is an experimental Clean-only extension of Clean-Peak v3. It has been
implemented and run on all 17 `legacy_smoke` orientations. The canonical
ACOM-matched image interface now passes the peak and orientation smoke tests.
No 1,081-pattern headline result is claimed yet.

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

## Two explicit Clean-E forward models

The canonical `acom_matched` mode and the diagnostic
`coherent_first_born` mode must not be mixed in one result table.

### Canonical `acom_matched`

`Crystal.generate_diffraction_pattern` supplies exactly the reflection
support, excitation-error envelope, and kinematical integrated intensity used
by the existing ACOM orientation-plan contract. Those reflections are then
converted into finite convergent-beam disks using subpixel placement,
oversampling, coherent accumulation where disks overlap, and detector pixel
integration.

This mode isolates the requested interface:

```text
matched kinematical reflections
→ diffraction image
→ automatic disk detection
→ same ACOM
```

It is not claimed to validate a new scattering model.

### Diagnostic `coherent_first_born`

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

The diagnostic model is more physical than drawing Gaussian points, but it is still an
ideal coherent first-Born model. With non-overlapping disks, zero aberration,
and no detector effects, its images will remain much cleaner and more regular
than experimental CBED. Its finite-thickness sinc sidelobes do not match the
current Gaussian excitation-error ACOM plan, so it is retained as a
model-generalization diagnostic rather than the canonical upper bound.

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

Generate the non-canonical First-Born diagnostic without overwriting the
matched files:

```bash
conda run -n or4d-clean \
  python scripts/02b_generate_clean_images.py \
  --role legacy_smoke \
  --forward-model coherent_first_born
```

`scripts/07_run_acom_baseline.py` accepts `--peak-file`,
`--prediction-file`, and `--report-tag`/`--output-tag`; every path uses the
same frozen 2° ACOM parameters. `scripts/15_run_clean_counted_acom.py` runs all
dose/seed/detector peak files and then aggregates them.

## Measured 17-case matched smoke results

Peak recovery against the current physical oracle:

| Input | Detector | Precision | Recall | RMSE (px) | P95 (px) |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean-E | AutoDisk | 1.000 | 1.000 | 0.406 | 0.620 |
| Clean-E | find_Bragg_disks | 1.000 | 1.000 | 0.011 | 0.017 |
| \(10^4\) e⁻ | AutoDisk | 0.568±0.008 | 0.510±0.006 | 0.562±0.009 | 0.909±0.024 |
| \(10^4\) e⁻ | find_Bragg_disks | 0.696±0.008 | 0.655±0.010 | 0.485±0.011 | 0.883±0.026 |
| \(10^5\) e⁻ | AutoDisk | 0.892±0.011 | 0.891±0.012 | 0.497±0.008 | 0.838±0.026 |
| \(10^5\) e⁻ | find_Bragg_disks | 0.969±0.008 | 0.968±0.008 | 0.304±0.013 | 0.650±0.052 |
| \(10^6\) e⁻ | AutoDisk | 0.997±0.001 | 0.997±0.001 | 0.426±0.001 | 0.665±0.005 |
| \(10^6\) e⁻ | find_Bragg_disks | 1.000±0.000 | 1.000±0.000 | 0.111±0.008 | 0.234±0.014 |

The five-repeat standard deviations above are real independent multinomial
realizations.

### Why the forward models are separated

A controlled 17-case ablation established that the initial First-Born failure
was a model-contract failure:

| ACOM input | Acc@2° | >5° |
| --- | ---: | ---: |
| Original ACOM coordinates and intensities | 88.2% | 2/17 |
| Original coordinates, uniform intensities | 88.2% | 2/17 |
| Original coordinates, First-Born image intensities | 47.1% | 4/17 |
| First-Born coordinates, uniform intensities | 29.4% | 10/17 |

The finite-thickness model contributes hundreds of sinc-sidelobe reflections
that the current Gaussian excitation-error orientation plan does not model.
Changing ACOM intensity exponents improved some cases but did not remove the
branch failures. The canonical matched mode therefore tests the image
interface honestly, while the First-Born mode requires its own matching
orientation plan before it can become a scored track.

ACOM results:

| Input | Acc@2° | Acc@5° | >5° |
| --- | ---: | ---: | ---: |
| Matched image oracle | 88.2% | 88.2% | 2/17 |
| Clean-E AutoDisk | 88.2% | 88.2% | 2/17 |
| Clean-E find_Bragg_disks | 88.2% | 88.2% | 2/17 |
| \(10^4\) e⁻ AutoDisk, five-seed mean | 72.9% | 88.2% | 2.0/17 |
| \(10^4\) e⁻ find_Bragg_disks, five-seed mean | 82.4% | 88.2% | 2.0/17 |
| \(10^5\) e⁻ AutoDisk, five-seed mean | 88.2% | 88.2% | 2.0/17 |
| \(10^5\) e⁻ find_Bragg_disks, five-seed mean | 88.2% | 88.2% | 2.0/17 |
| \(10^6\) e⁻ AutoDisk, five-seed mean | 88.2% | 88.2% | 2.0/17 |
| \(10^6\) e⁻ find_Bragg_disks, five-seed mean | 88.2% | 88.2% | 2.0/17 |

The two >5° cases are the already documented approximately 72° `[001]`
ambiguity in `clean_ori_000` and `clean_ori_005`; the image pipeline introduces
no new catastrophic errors at \(10^5\) or \(10^6\) electrons.

On the 15 samples where oracle ACOM is within 5° of ground truth, Clean-E
orientation differences from the oracle have P95 0.047° for AutoDisk and
0.002° for `find_Bragg_disks`.

## Next gates

1. Freeze the 17 legacy cases as the detector calibration split; do not tune
   thresholds on `headline_core`.
2. Run the canonical matched path on all 1,081 patterns and report the 1,024
   headline cases separately from the 40 grid probes.
3. Keep `coherent_first_born` results in a separate model-generalization
   report until a matching physical orientation plan is implemented.
4. Add detector background, PSF, and direct-beam-fraction sweeps only after the
   matched full run is frozen.
