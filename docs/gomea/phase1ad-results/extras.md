# Phase extras

Result directories: `../gomea-results/phase1ad/phase1ad-sr`, `../gomea-results/phase1ad/phase1ad-sr-Con`, `../gomea-results/phase1ad/phase1ad-sr-ext-EnC`, `../gomea-results/phase1ad/phase1ad-sr-ext-EnH`

## Regression (medians per arm)

| Problem | Arm | Runs | Not ok | Test NMSE | Train NMSE | Test − train | Test / train | Size (nodes) | CPU per run (s) |
|---|---|---|---|---|---|---|---|---|---|
| Con | GOMEA-UNI | 10 | 0 | 29.2 | 24.2 | 4.4 | 1.19 | 309.0 | 2,492 |
| Con | GOMEA-UNI-PEN | 10 | 0 | 37.2 | 33.8 | 3.8 | 1.12 | 92.5 | 1,028 |
| Con | GOMEA-UNI-S10 | 10 | 0 | 43.2 | 37.5 | 2.7 | 1.08 | 29.5 | 494 |
| EnC | GOMEA | 20 | 0 | 19.2 | 17.1 | 2.2 | 1.13 | 370.5 | 3,255 |
| EnC | GOMEA-UNI | 20 | 0 | 15.7 | 13.7 | 2.4 | 1.18 | 731.5 | 5,834 |
| EnH | GOMEA | 20 | 0 | 15.6 | 14.3 | 1.5 | 1.13 | 408.5 | 3,192 |
| EnH | GOMEA-UNI | 20 | 0 | 12.8 | 11.3 | 1.5 | 1.15 | 707.5 | 5,929 |

## GOMEA generations per deme

No per-generation telemetry (no `events.jsonl`) for: Con:GOMEA-UNI, Con:GOMEA-UNI-PEN, Con:GOMEA-UNI-S10, EnC:GOMEA, EnC:GOMEA-UNI, EnH:GOMEA, EnH:GOMEA-UNI.

