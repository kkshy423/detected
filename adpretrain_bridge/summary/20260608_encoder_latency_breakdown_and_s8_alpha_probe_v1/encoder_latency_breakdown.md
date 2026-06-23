# Encoder latency breakdown

Diagnostic only. Same encoder, same predecoded CPU/PIL input tensors, batch=1, test images, seed=0.

## Backbone meta

- requested backbone: `dinov2-large`
- encoder: `DinoModel` / underlying ``
- blocks/embed_dim/heads/patch:  /  /  / 
- input shape: `[1, 3, 224, 224]`
- output shapes: `[[1, 1024, 16, 16], [1, 1024, 16, 16], [1, 1024, 16, 16], [1, 1024, 16, 16]]`
- param count: 304.369 M

## Timing table

| tf32 | timing | median_or_amortized_ms | mean_ms | p90_ms | p95_ms | launch_median_ms |
|---|---|---:|---:|---:|---:|---:|
| off | synced | 18.2257 | 18.2350 | 18.4437 | 18.4683 |  |
| off | pure | 18.1474 | 18.1474 |  |  | 18.1270 |
| on | synced | 11.2931 | 11.3459 | 11.3176 | 11.3301 |  |
| on | pure | 11.1467 | 11.1467 |  |  | 11.1429 |

## Attribution

- TF32 OFF synced median: 18.2257 ms; TF32 ON synced median: 11.2931 ms; speedup=1.61x.
- Sync/timing gap under TF32 OFF: 0.0783 ms (synced median - pure amortized).
- Sync/timing gap under TF32 ON: 0.1463 ms (synced median - pure amortized).
- Pure timing is total synchronized wall time divided by N images; launch_median_ms is shown only as CPU launch reference.
- This experiment does not decide whether TF32 should be enabled in production; TF32 would need a separate numeric-equivalence check.
