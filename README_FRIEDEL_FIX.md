# Clean-Peak ACOM evaluation and visualization fix

This patch makes four changes:

1. Proper point-group operations are selected by checking the determinant of the raw pymatgen operation before orthogonalization.
2. Clean-Peak evaluation reports both strict crystal misorientation and Friedel-equivalent misorientation.
3. The Orientation Plan audit includes the ACOM normal and mirror search branches and evaluates off-grid distance in the same Friedel-equivalence space as Clean-Peak scoring.
4. Three figures are generated:
   - `reports/acom_error_comparison.png`
   - `reports/acom_offgrid_vs_error.png`
   - `reports/acom_peak_overlay.png`

The benchmark submission format remains unchanged.
