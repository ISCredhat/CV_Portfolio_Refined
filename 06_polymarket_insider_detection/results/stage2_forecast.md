# Stage 2, task C: forecasting benchmark against the market price (the stop gate)

Aggregate results only. Frozen rule (docs/STAGE2_PLAN.md section 9): a model's out-of-sample log-loss below the price benchmark in BOTH halves of the tune period (end_ts Jan-Mar and Apr-Jun 2026) with the lower 95% event-cluster bootstrap bound of the improvement above zero, for at least one horizon. The sealed period was never loaded.

## Verdict: FAIL: no measurable information beyond the price. The strategy track STOPS (S2/S3/S5 not run).

Improvement = mean(price loss - model loss); positive means the model beats the price. Prices clipped to [0.001, 0.999].

## Sample

| horizon | all snapshots (>= 20 trades, open at t) | test rows | markets | events | Jan-Mar | Apr-Jun | YES rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7d | 6,293 | 2,936 | 2,936 | 647 | 1,253 | 1,683 | 17.2% |
| 3d | 7,072 | 2,944 | 2,944 | 659 | 1,257 | 1,687 | 17.3% |
| 1d | 7,237 | 2,908 | 2,908 | 662 | 1,235 | 1,673 | 17.3% |

## Out-of-sample log-loss and Brier: price vs models, with improvement and 95% event-cluster interval

| horizon | model | sample | n | price LL | model LL | LL improvement [95%] | Brier improvement [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7d | recal | JanMar | 1,253 | 0.1703 | 0.1712 | -0.00085 [-0.00248, +0.00105] | -0.00034 [-0.00110, +0.00050] |
| 7d | recal | AprJun | 1,683 | 0.1816 | 0.1817 | -0.00004 [-0.00143, +0.00128] | -0.00007 [-0.00054, +0.00040] |
| 7d | recal | pooled | 2,936 | 0.1768 | 0.1772 | -0.00039 [-0.00150, +0.00074] | -0.00018 [-0.00065, +0.00023] |
| 7d | hgb_all | JanMar | 1,253 | 0.1703 | 0.1829 | -0.01260 [-0.02117, -0.00394] | -0.00437 [-0.00730, -0.00174] |
| 7d | hgb_all | AprJun | 1,683 | 0.1816 | 0.1920 | -0.01039 [-0.01612, -0.00450] | -0.00380 [-0.00587, -0.00170] |
| 7d | hgb_all | pooled | 2,936 | 0.1768 | 0.1881 | -0.01133 [-0.01655, -0.00659] | -0.00404 [-0.00592, -0.00236] |
| 3d | recal | JanMar | 1,257 | 0.1559 | 0.1569 | -0.00096 [-0.00309, +0.00124] | -0.00001 [-0.00094, +0.00098] |
| 3d | recal | AprJun | 1,687 | 0.1671 | 0.1669 | +0.00023 [-0.00124, +0.00179] | +0.00001 [-0.00061, +0.00066] |
| 3d | recal | pooled | 2,944 | 0.1623 | 0.1626 | -0.00028 [-0.00155, +0.00110] | -0.00000 [-0.00056, +0.00060] |
| 3d | hgb_all | JanMar | 1,257 | 0.1559 | 0.1614 | -0.00549 [-0.01253, +0.00147] | -0.00156 [-0.00403, +0.00065] |
| 3d | hgb_all | AprJun | 1,687 | 0.1671 | 0.1735 | -0.00643 [-0.01364, -0.00019] | -0.00226 [-0.00475, -0.00016] |
| 3d | hgb_all | pooled | 2,944 | 0.1623 | 0.1684 | -0.00603 [-0.01063, -0.00122] | -0.00196 [-0.00361, -0.00025] |
| 1d | recal | JanMar | 1,235 | 0.1536 | 0.1542 | -0.00059 [-0.00238, +0.00125] | -0.00002 [-0.00076, +0.00080] |
| 1d | recal | AprJun | 1,673 | 0.1597 | 0.1596 | +0.00007 [-0.00113, +0.00129] | +0.00007 [-0.00038, +0.00053] |
| 1d | recal | pooled | 2,908 | 0.1571 | 0.1573 | -0.00021 [-0.00122, +0.00083] | +0.00003 [-0.00036, +0.00046] |
| 1d | hgb_all | JanMar | 1,235 | 0.1536 | 0.1608 | -0.00714 [-0.01434, -0.00053] | -0.00140 [-0.00323, +0.00043] |
| 1d | hgb_all | AprJun | 1,673 | 0.1597 | 0.1664 | -0.00673 [-0.01234, -0.00136] | -0.00108 [-0.00266, +0.00052] |
| 1d | hgb_all | pooled | 2,908 | 0.1571 | 0.1640 | -0.00690 [-0.01150, -0.00275] | -0.00121 [-0.00244, -0.00004] |

## Pass rule by horizon and model

| horizon | recalibration passes | feature model (HGB) passes |
| --- | --- | --- |
| 7d | fail | fail |
| 3d | fail | fail |
| 1d | fail | fail |

Six horizon-by-model comparisons are made, uncorrected for multiplicity (the frozen rule is 'at least one horizon'); a marginal pass should be read with that in mind. A pass by the price-only recalibration would show calibration, not information beyond the price.

## Feature-group ablation (pooled test period; log-loss improvement vs price, HGB depth 3 / 200 trees / lr 0.05)

'plus' = logit price plus that group only; 'minus' = all features without that group.

| horizon | model | LL improvement [95%] | Brier improvement |
| --- | --- | --- | --- |
| 7d | hgb_price_only | -0.00254 [-0.00581, +0.00037] | -0.00144 |
| 7d | hgb_plus_price_dynamics | -0.00459 [-0.00898, -0.00069] | -0.00189 |
| 7d | hgb_plus_flow | -0.00927 [-0.01325, -0.00534] | -0.00349 |
| 7d | hgb_plus_wallet_skill | -0.00258 [-0.00611, +0.00088] | -0.00158 |
| 7d | hgb_plus_impact_other | -0.00925 [-0.01432, -0.00448] | -0.00347 |
| 7d | hgb_minus_price_dynamics | -0.01277 [-0.01767, -0.00791] | -0.00469 |
| 7d | hgb_minus_flow | -0.00711 [-0.01148, -0.00295] | -0.00269 |
| 7d | hgb_minus_wallet_skill | -0.01112 [-0.01641, -0.00624] | -0.00379 |
| 7d | hgb_minus_impact_other | -0.00799 [-0.01225, -0.00413] | -0.00284 |
| 7d | hgb_all | -0.01133 [-0.01655, -0.00659] | -0.00404 |
| 3d | hgb_price_only | -0.00267 [-0.00543, +0.00000] | -0.00106 |
| 3d | hgb_plus_price_dynamics | -0.00470 [-0.00834, -0.00103] | -0.00137 |
| 3d | hgb_plus_flow | -0.00604 [-0.01085, -0.00188] | -0.00160 |
| 3d | hgb_plus_wallet_skill | -0.00469 [-0.00771, -0.00176] | -0.00176 |
| 3d | hgb_plus_impact_other | -0.00659 [-0.01136, -0.00187] | -0.00197 |
| 3d | hgb_minus_price_dynamics | -0.00824 [-0.01318, -0.00383] | -0.00259 |
| 3d | hgb_minus_flow | -0.00505 [-0.00917, -0.00051] | -0.00166 |
| 3d | hgb_minus_wallet_skill | -0.00628 [-0.01104, -0.00129] | -0.00188 |
| 3d | hgb_minus_impact_other | -0.00426 [-0.00835, -0.00044] | -0.00107 |
| 3d | hgb_all | -0.00603 [-0.01063, -0.00122] | -0.00196 |
| 1d | hgb_price_only | -0.00565 [-0.00993, -0.00196] | -0.00064 |
| 1d | hgb_plus_price_dynamics | -0.00286 [-0.00700, +0.00109] | -0.00024 |
| 1d | hgb_plus_flow | -0.00872 [-0.01322, -0.00437] | -0.00197 |
| 1d | hgb_plus_wallet_skill | -0.00517 [-0.00891, -0.00187] | -0.00092 |
| 1d | hgb_plus_impact_other | -0.00776 [-0.01196, -0.00381] | -0.00158 |
| 1d | hgb_minus_price_dynamics | -0.00892 [-0.01370, -0.00431] | -0.00194 |
| 1d | hgb_minus_flow | -0.00491 [-0.00881, -0.00098] | -0.00064 |
| 1d | hgb_minus_wallet_skill | -0.00726 [-0.01140, -0.00335] | -0.00154 |
| 1d | hgb_minus_impact_other | -0.00563 [-0.00939, -0.00224] | -0.00099 |
| 1d | hgb_all | -0.00690 [-0.01150, -0.00275] | -0.00121 |

## Stress test: dropping the largest 10% of test markets by volume to date

| horizon | model | sample | LL improvement [95%] |
| --- | --- | --- | --- |
| 7d | recal | JanMar | -0.00118 [-0.00323, +0.00095] (n 1,145) |
| 7d | recal | AprJun | -0.00013 [-0.00160, +0.00128] (n 1,497) |
| 7d | recal | pooled | -0.00058 [-0.00176, +0.00070] (n 2,642) |
| 7d | hgb_all | JanMar | -0.01484 [-0.02466, -0.00563] (n 1,145) |
| 7d | hgb_all | AprJun | -0.01075 [-0.01707, -0.00476] (n 1,497) |
| 7d | hgb_all | pooled | -0.01252 [-0.01773, -0.00715] (n 2,642) |
| 3d | recal | JanMar | -0.00096 [-0.00337, +0.00166] (n 1,138) |
| 3d | recal | AprJun | +0.00010 [-0.00160, +0.00174] (n 1,511) |
| 3d | recal | pooled | -0.00035 [-0.00191, +0.00103] (n 2,649) |
| 3d | hgb_all | JanMar | -0.00528 [-0.01313, +0.00232] (n 1,138) |
| 3d | hgb_all | AprJun | -0.00770 [-0.01519, -0.00108] (n 1,511) |
| 3d | hgb_all | pooled | -0.00666 [-0.01265, -0.00140] (n 2,649) |
| 1d | recal | JanMar | -0.00060 [-0.00251, +0.00145] (n 1,124) |
| 1d | recal | AprJun | -0.00019 [-0.00151, +0.00119] (n 1,493) |
| 1d | recal | pooled | -0.00036 [-0.00146, +0.00087] (n 2,617) |
| 1d | hgb_all | JanMar | -0.00788 [-0.01551, -0.00075] (n 1,124) |
| 1d | hgb_all | AprJun | -0.00723 [-0.01283, -0.00133] (n 1,493) |
| 1d | hgb_all | pooled | -0.00751 [-0.01241, -0.00277] (n 2,617) |

## By information type (pooled test period)

| horizon | info type | model | n | LL improvement [95%] |
| --- | --- | --- | --- | --- |
| 7d | deadline | recal | 612 | -0.00046 [-0.00347, +0.00218] |
| 7d | deadline | hgb_all | 612 | -0.00811 [-0.01945, +0.00136] |
| 7d | scheduled_reveal | recal | 2,028 | -0.00036 [-0.00140, +0.00077] |
| 7d | scheduled_reveal | hgb_all | 2,028 | -0.01057 [-0.01666, -0.00500] |
| 7d | unscheduled_event | recal | 296 | -0.00043 [-0.00506, +0.00384] |
| 7d | unscheduled_event | hgb_all | 296 | -0.02320 [-0.04582, -0.00603] |
| 3d | deadline | recal | 620 | -0.00022 [-0.00281, +0.00229] |
| 3d | deadline | hgb_all | 620 | -0.00420 [-0.01543, +0.00621] |
| 3d | scheduled_reveal | recal | 2,033 | -0.00017 [-0.00164, +0.00149] |
| 3d | scheduled_reveal | hgb_all | 2,033 | -0.00567 [-0.01185, -0.00029] |
| 3d | unscheduled_event | recal | 291 | -0.00113 [-0.00657, +0.00336] |
| 3d | unscheduled_event | hgb_all | 291 | -0.01247 [-0.02474, -0.00134] |
| 1d | deadline | recal | 584 | -0.00142 [-0.00350, +0.00046] |
| 1d | deadline | hgb_all | 584 | -0.00569 [-0.01574, +0.00420] |
| 1d | scheduled_reveal | recal | 2,043 | +0.00036 [-0.00083, +0.00156] |
| 1d | scheduled_reveal | hgb_all | 2,043 | -0.00720 [-0.01265, -0.00207] |
| 1d | unscheduled_event | recal | 281 | -0.00182 [-0.00595, +0.00141] |
| 1d | unscheduled_event | hgb_all | 281 | -0.00724 [-0.01618, +0.00157] |

## Calibration of the price benchmark (pooled test rows, quantile bins)

| horizon | bin | n | mean price | outcome rate |
| --- | --- | --- | --- | --- |
| 7d | 1 | 250 | 0.001 | 0.000 |
| 7d | 2 | 312 | 0.001 | 0.000 |
| 7d | 3 | 318 | 0.003 | 0.006 |
| 7d | 4 | 289 | 0.006 | 0.003 |
| 7d | 5 | 297 | 0.014 | 0.007 |
| 7d | 6 | 293 | 0.033 | 0.010 |
| 7d | 7 | 292 | 0.077 | 0.086 |
| 7d | 8 | 297 | 0.195 | 0.168 |
| 7d | 9 | 294 | 0.514 | 0.500 |
| 7d | 10 | 294 | 0.911 | 0.939 |
| 3d | 1 | 398 | 0.001 | 0.000 |
| 3d | 2 | 376 | 0.001 | 0.005 |
| 3d | 3 | 400 | 0.003 | 0.000 |
| 3d | 4 | 294 | 0.009 | 0.010 |
| 3d | 5 | 298 | 0.023 | 0.017 |
| 3d | 6 | 295 | 0.061 | 0.047 |
| 3d | 7 | 294 | 0.181 | 0.167 |
| 3d | 8 | 292 | 0.544 | 0.521 |
| 3d | 9 | 297 | 0.940 | 0.956 |
| 1d | 1 | 470 | 0.001 | 0.000 |
| 1d | 2 | 387 | 0.001 | 0.000 |
| 1d | 3 | 306 | 0.002 | 0.007 |
| 1d | 4 | 266 | 0.006 | 0.008 |
| 1d | 5 | 316 | 0.015 | 0.016 |
| 1d | 6 | 287 | 0.045 | 0.042 |
| 1d | 7 | 294 | 0.165 | 0.163 |
| 1d | 8 | 288 | 0.562 | 0.538 |
| 1d | 9 | 294 | 0.955 | 0.952 |

## Construction and caveats

- Snapshot t = end_ts - H; markets with closed_ts <= t, fewer than 20 taker trades before t, or an unknown YES/NO outcome are excluded. Selection on 'still open at t' uses only information knowable at t.
- Test rows: market closed in the tune period (closed_ts 2026-01-01 to 2026-06-30), end_ts month Jan to Jun 2026, t >= 2026-01-01. Training rows for test month M: closed_ts at least 7 days before the start of M (labels known), fit period and earlier tune months. Markets whose end_ts falls in the tune window but that closed after 2026-06-30 are sealed and excluded, which selects on survival to the cut-off.
- Wallet-skill features: fit-period 24 h markout classification (analytic sign-randomisation p with Benjamini-Hochberg 5% and EB shrunk skill > 0); for fit-period training rows the classification is cross-fitted by event so a market never feeds its own feature. The classification uses markouts of other markets that can be later than t for training rows only (never for test rows).
- 'Buying wallets' in the Herfindahl are the wallets pushing the YES price up (buy Yes or sell No); the original BUY/SELL flag is not used because the Yes/No sides are folded to YES terms.
- Hours to end equals H by construction (a constant within a horizon).
- Price history is the stored winner-token history plus taker fills; no order-book data, so the price is a last-trade price, not a mid.
