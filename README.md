# Universal VE Simulator (ECLIPSE + SPE10/SPE102)

Streamlit app that runs a VE + Darcy + Land forward simulator on:
- **NPZ**: `phi` and `k` 2D arrays
- **ECLIPSE ZIP**: `*.EGRID + *.INIT` (PORO, PERMX, ACTNUM...) via `resdata`
- **SPE10/SPE102 ZIP**: `spe_phi.dat` and `spe_perm.dat` (KX,KY,KZ stacked)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Schedule CSV
Upload a CSV with columns:
- `t` : schedule points (can be sparse)
- `q` : signed rate

The app resamples to uniform timesteps 0..Nt-1.

## Notes
SPE10 dims default to 60x220x85. If you use a different SPE variant, edit `spe10_loader.py`.
