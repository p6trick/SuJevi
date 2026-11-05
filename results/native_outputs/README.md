# Native model outputs

This directory stores responses exactly in each model's native schema. Do not
replace native probability fields with the future universal routing interface.

`kev-4b.json` is created by `scripts/run_kev_native.py` from the official Kev
Python loader, without an HTTP server. `kev-4b.probabilities.json` keeps the
complete distribution for every question, including both `false` and `true`
for Noul. Neither is considered successful until the user-provided execution
log is reviewed.
