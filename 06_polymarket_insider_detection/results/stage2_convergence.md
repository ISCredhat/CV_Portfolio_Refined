# Stage 2A: tradable convergence (strategy 2)

Frozen definition and pass rule: docs/STAGE2_PLAN.md section 9. Research only, not advice. Fit and tune periods only (sealed period not opened). Aggregate results, no wallets, no market ids.

## Events

Eligible events: 535

- fit: 249 events (3-4: 33, 5-9: 158, 10+: 58)
- tune: 286 events (3-4: 12, 5-9: 31, 10+: 243)

Exclusions: events_total=1201, no_independent_time=0, sibling_without_close_time=9, sealed_period_dropped=478, not_neg_risk=179, fewer_than_3_outcomes=0, events_eligible=535

## Trades skipped by reason (event-horizon-rule-delay attempts)

| horizon | rule | delay | reason | count |
|---|---|---|---|---|
| 48h | all | all | fewer_than_3_comparable_outcomes | 29 |
| 48h | highest_price | 1 | no_price_within_6h | 160 |
| 48h | highest_price | 1 | volume_cap_below_min | 46 |
| 48h | highest_price | 30 | no_price_within_6h | 157 |
| 48h | highest_price | 30 | volume_cap_below_min | 48 |
| 48h | highest_price | 5 | no_price_within_6h | 159 |
| 48h | highest_price | 5 | volume_cap_below_min | 46 |
| 48h | logodds_rise | 1 | no_price_within_6h | 198 |
| 48h | logodds_rise | 1 | volume_cap_below_min | 51 |
| 48h | logodds_rise | 30 | no_price_within_6h | 195 |
| 48h | logodds_rise | 30 | volume_cap_below_min | 53 |
| 48h | logodds_rise | 5 | no_price_within_6h | 198 |
| 48h | logodds_rise | 5 | volume_cap_below_min | 51 |
| 72h | all | all | fewer_than_3_comparable_outcomes | 56 |
| 72h | highest_price | 1 | no_price_within_6h | 178 |
| 72h | highest_price | 1 | volume_cap_below_min | 61 |
| 72h | highest_price | 30 | no_price_within_6h | 179 |
| 72h | highest_price | 30 | volume_cap_below_min | 55 |
| 72h | highest_price | 5 | no_price_within_6h | 179 |
| 72h | highest_price | 5 | volume_cap_below_min | 60 |
| 72h | logodds_rise | 1 | no_price_within_6h | 216 |
| 72h | logodds_rise | 1 | volume_cap_below_min | 58 |
| 72h | logodds_rise | 30 | no_price_within_6h | 221 |
| 72h | logodds_rise | 30 | volume_cap_below_min | 53 |
| 72h | logodds_rise | 5 | no_price_within_6h | 218 |
| 72h | logodds_rise | 5 | volume_cap_below_min | 57 |
| 7d | all | all | fewer_than_3_comparable_outcomes | 106 |
| 7d | highest_price | 1 | no_price_within_6h | 194 |
| 7d | highest_price | 1 | volume_cap_below_min | 65 |
| 7d | highest_price | 30 | no_price_within_6h | 199 |
| 7d | highest_price | 30 | volume_cap_below_min | 66 |
| 7d | highest_price | 5 | no_price_within_6h | 194 |
| 7d | highest_price | 5 | volume_cap_below_min | 65 |
| 7d | logodds_rise | 1 | no_price_within_6h | 237 |
| 7d | logodds_rise | 1 | volume_cap_below_min | 64 |
| 7d | logodds_rise | 30 | no_price_within_6h | 240 |
| 7d | logodds_rise | 30 | volume_cap_below_min | 63 |
| 7d | logodds_rise | 5 | no_price_within_6h | 237 |
| 7d | logodds_rise | 5 | volume_cap_below_min | 64 |

Trades at the 5-min headline delay: 1300; stake scaled down by the 1% volume cap: 90.9%; entry from the snapshot price series (sibling with no taker tape): 0.9%.

## Pass table (frozen rule: lower 95% bound > 0, beats placebo, positive at fee x2 + 30-min entry)

| rule | horizon | lo95>0 | beats placebo | stress>0 | PASS | one-sided p | Bonferroni p (6 cells) |
|---|---|---|---|---|---|---|---|
| logodds_rise | 7d | False | False | False | FAIL | 0.972 | 1.000 |
| logodds_rise | 72h | False | False | False | FAIL | 0.871 | 1.000 |
| logodds_rise | 48h | False | False | False | FAIL | 0.998 | 1.000 |
| highest_price | 7d | False | True | True | FAIL | 0.275 | 1.000 |
| highest_price | 72h | False | False | False | FAIL | 0.554 | 1.000 |
| highest_price | 48h | False | False | False | FAIL | 0.539 | 1.000 |

Cells tested: 6. Passing the frozen rule: 0. Passing after Bonferroni over 6 cells: 0. By chance alone at a one-sided 2.5% level about 0.15 cells (0.30 at 5%) are expected to show a positive lower bound. 'Beats placebo' is read as: leaders' net return on the matched trades exceeds the upper 95% limit of the 200 placebo draws.

## Results (net return per dollar, 95% event-bootstrap interval)

| group | value | rule | horizon | delay | cost x | events | trades | win rate | mean entry | win-entry [95%] | net ret [95%] | placebo mean [95%] | leaders on matched | matched trades | median stake |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | all | logodds_rise | 7d | 5 | 1 | 128 | 128 | 0.594 | 0.564 | 0.030 [-0.017, 0.080] | -0.176 [-0.369, 0.005] | -0.193 [-0.494, 2.633] | -0.805 | 22 | 7.0 |
| period | fit | logodds_rise | 7d | 5 | 1 | 57 | 57 | 0.561 | 0.557 | 0.005 [-0.068, 0.081] | -0.221 [-0.454, 0.002] | -0.416 [-0.435, -0.399] | -0.798 | 11 | 15.6 |
| period | tune | logodds_rise | 7d | 5 | 1 | 71 | 71 | 0.620 | 0.569 | 0.050 [-0.016, 0.118] | -0.114 [-0.415, 0.206] | 0.174 [-0.550, 5.860] | -0.810 | 11 | 5.1 |
| size | 3-4 | logodds_rise | 7d | 5 | 1 | 15 | 15 | 0.467 | 0.569 | -0.102 [-0.274, 0.045] | -0.335 [-0.747, 0.028] | -0.339 [-0.339, -0.339] | -1.000 | 3 | 16.8 |
| size | 5-9 | logodds_rise | 7d | 5 | 1 | 42 | 42 | 0.524 | 0.490 | 0.034 [-0.072, 0.133] | -0.174 [-0.488, 0.164] | -0.437 [-0.485, -0.358] | -0.786 | 11 | 9.7 |
| size | 10+ | logodds_rise | 7d | 5 | 1 | 71 | 71 | 0.662 | 0.607 | 0.055 [-0.000, 0.112] | -0.102 [-0.364, 0.182] | 0.231 [-0.571, 7.799] | -0.706 | 8 | 5.7 |
| tape_only | tape_only | logodds_rise | 7d | 5 | 1 | 126 | 126 | 0.603 | 0.572 | 0.032 [-0.016, 0.081] | -0.110 [-0.302, 0.077] | -0.003 [-0.375, 1.031] | -0.769 | 21 | 6.9 |
| overall | all | logodds_rise | 7d | 1 | 1 | 128 | 128 | 0.586 | 0.563 | 0.023 [-0.029, 0.074] | -0.180 [-0.367, 0.005] | -0.201 [-0.494, 2.381] | -0.804 | 22 | 7.0 |
| overall | all | logodds_rise | 7d | 30 | 1 | 126 | 126 | 0.603 | 0.572 | 0.032 [-0.017, 0.086] | -0.137 [-0.335, 0.058] | -0.194 [-0.709, 3.260] | -0.758 | 19 | 7.0 |
| stress | fee_x2_delay30 | logodds_rise | 7d | 30 | 2 | 126 | 126 | 0.603 | 0.572 | 0.032 [-0.017, 0.086] | -0.147 [-0.340, 0.045] | -0.194 [-0.709, 3.260] | -0.758 | 19 | 7.0 |
| overall | all | logodds_rise | 72h | 5 | 1 | 204 | 204 | 0.613 | 0.570 | 0.043 [0.001, 0.086] | -0.091 [-0.227, 0.058] | -0.558 [-0.808, 3.244] | -0.664 | 38 | 9.9 |
| period | fit | logodds_rise | 72h | 5 | 1 | 108 | 108 | 0.583 | 0.525 | 0.058 [0.001, 0.122] | -0.128 [-0.283, 0.041] | -0.545 [-0.731, 0.020] | -0.980 | 21 | 14.1 |
| period | tune | logodds_rise | 72h | 5 | 1 | 96 | 96 | 0.646 | 0.621 | 0.025 [-0.046, 0.090] | -0.032 [-0.262, 0.194] | -0.191 [-0.986, 11.295] | 0.008 | 17 | 7.7 |
| size | 3-4 | logodds_rise | 72h | 5 | 1 | 22 | 22 | 0.727 | 0.625 | 0.102 [0.007, 0.219] | 0.010 [-0.184, 0.144] | -0.597 [-0.597, -0.597] | 0.012 | 3 | 14.2 |
| size | 5-9 | logodds_rise | 72h | 5 | 1 | 80 | 80 | 0.575 | 0.524 | 0.051 [-0.037, 0.130] | 0.018 [-0.225, 0.273] | -0.308 [-0.416, -0.155] | 0.017 | 15 | 10.6 |
| size | 10+ | logodds_rise | 72h | 5 | 1 | 102 | 102 | 0.618 | 0.595 | 0.023 [-0.036, 0.079] | -0.198 [-0.382, -0.013] | -0.351 [-1.000, 5.981] | -1.000 | 20 | 8.7 |
| tape_only | tape_only | logodds_rise | 72h | 5 | 1 | 203 | 203 | 0.616 | 0.573 | 0.043 [-0.003, 0.084] | -0.073 [-0.211, 0.070] | -0.472 [-0.790, 3.935] | -0.615 | 37 | 9.9 |
| overall | all | logodds_rise | 72h | 1 | 1 | 205 | 205 | 0.615 | 0.572 | 0.043 [-0.004, 0.085] | -0.093 [-0.225, 0.048] | -0.569 [-0.829, 3.258] | -0.673 | 38 | 9.9 |
| overall | all | logodds_rise | 72h | 30 | 1 | 205 | 205 | 0.610 | 0.570 | 0.040 [-0.007, 0.081] | -0.079 [-0.212, 0.052] | -0.344 [-0.742, 4.312] | -0.612 | 39 | 9.9 |
| stress | fee_x2_delay30 | logodds_rise | 72h | 30 | 2 | 205 | 205 | 0.610 | 0.570 | 0.040 [-0.007, 0.081] | -0.089 [-0.220, 0.039] | -0.344 [-0.742, 4.312] | -0.612 | 39 | 9.9 |
| overall | all | logodds_rise | 48h | 5 | 1 | 257 | 257 | 0.553 | 0.566 | -0.013 [-0.046, 0.022] | -0.177 [-0.294, -0.054] | 0.178 [-0.460, 3.458] | -0.700 | 57 | 10.0 |
| period | fit | logodds_rise | 48h | 5 | 1 | 147 | 147 | 0.503 | 0.508 | -0.004 [-0.055, 0.046] | -0.151 [-0.324, 0.020] | 0.619 [-0.234, 6.113] | -0.864 | 36 | 11.8 |
| period | tune | logodds_rise | 48h | 5 | 1 | 110 | 110 | 0.618 | 0.643 | -0.025 [-0.078, 0.020] | -0.224 [-0.411, -0.043] | -0.411 [-1.000, 8.416] | -0.467 | 21 | 9.5 |
| size | 3-4 | logodds_rise | 48h | 5 | 1 | 33 | 33 | 0.667 | 0.653 | 0.013 [-0.057, 0.091] | -0.053 [-0.364, 0.243] | -1.000 [-1.000, -1.000] | -1.000 | 2 | 19.4 |
| size | 5-9 | logodds_rise | 48h | 5 | 1 | 106 | 106 | 0.481 | 0.500 | -0.019 [-0.077, 0.042] | -0.133 [-0.336, 0.041] | -0.150 [-0.524, 0.249] | -0.331 | 27 | 8.7 |
| size | 10+ | logodds_rise | 48h | 5 | 1 | 118 | 118 | 0.585 | 0.600 | -0.015 [-0.061, 0.030] | -0.257 [-0.454, -0.028] | 0.828 [-0.405, 7.931] | -0.949 | 28 | 10.2 |
| tape_only | tape_only | logodds_rise | 48h | 5 | 1 | 253 | 253 | 0.557 | 0.568 | -0.011 [-0.046, 0.024] | -0.145 [-0.274, -0.002] | 0.171 [-0.626, 4.082] | -0.633 | 55 | 10.0 |
| overall | all | logodds_rise | 48h | 1 | 1 | 257 | 257 | 0.553 | 0.566 | -0.013 [-0.046, 0.020] | -0.174 [-0.295, -0.041] | 0.282 [-0.455, 3.533] | -0.704 | 56 | 10.0 |
| overall | all | logodds_rise | 48h | 30 | 1 | 258 | 258 | 0.547 | 0.565 | -0.018 [-0.053, 0.016] | -0.180 [-0.297, -0.042] | 0.824 [-0.363, 4.332] | -0.692 | 58 | 9.9 |
| stress | fee_x2_delay30 | logodds_rise | 48h | 30 | 2 | 258 | 258 | 0.547 | 0.565 | -0.018 [-0.053, 0.016] | -0.188 [-0.302, -0.053] | 0.824 [-0.363, 4.332] | -0.692 | 58 | 9.9 |
| overall | all | highest_price | 7d | 5 | 1 | 170 | 170 | 0.735 | 0.727 | 0.009 [-0.050, 0.060] | 0.041 [-0.083, 0.162] | -0.560 [-0.610, -0.509] | 0.624 | 9 | 6.5 |
| period | fit | highest_price | 7d | 5 | 1 | 75 | 75 | 0.667 | 0.709 | -0.042 [-0.121, 0.042] | -0.049 [-0.244, 0.130] | -0.710 [-0.710, -0.710] | 0.632 | 3 | 8.6 |
| period | tune | highest_price | 7d | 5 | 1 | 95 | 95 | 0.789 | 0.741 | 0.048 [-0.022, 0.118] | 0.167 [0.011, 0.320] | -0.371 [-0.518, -0.199] | 0.610 | 6 | 5.8 |
| size | 3-4 | highest_price | 7d | 5 | 1 | 15 | 15 | 0.733 | 0.791 | -0.058 [-0.237, 0.102] | -0.119 [-0.446, 0.090] | 0.843 [0.843, 0.843] | -1.000 | 1 | 18.2 |
| size | 5-9 | highest_price | 7d | 5 | 1 | 58 | 58 | 0.603 | 0.624 | -0.021 [-0.139, 0.093] | 0.089 [-0.138, 0.323] | -0.764 [-0.802, -0.721] | 0.859 | 6 | 6.1 |
| size | 10+ | highest_price | 7d | 5 | 1 | 97 | 97 | 0.814 | 0.778 | 0.036 [-0.031, 0.096] | 0.068 [-0.112, 0.229] | -0.692 [-0.692, -0.692] | 0.930 | 2 | 6.4 |
| tape_only | tape_only | highest_price | 7d | 5 | 1 | 169 | 169 | 0.734 | 0.726 | 0.008 [-0.050, 0.064] | 0.038 [-0.091, 0.170] | -0.560 [-0.610, -0.509] | 0.624 | 9 | 6.4 |
| overall | all | highest_price | 7d | 1 | 1 | 170 | 170 | 0.729 | 0.726 | 0.004 [-0.055, 0.056] | 0.040 [-0.086, 0.163] | -0.561 [-0.611, -0.510] | 0.636 | 9 | 6.5 |
| overall | all | highest_price | 7d | 30 | 1 | 164 | 164 | 0.756 | 0.735 | 0.021 [-0.035, 0.076] | 0.066 [-0.083, 0.217] | -0.334 [-0.476, -0.275] | 0.533 | 8 | 7.1 |
| stress | fee_x2_delay30 | highest_price | 7d | 30 | 2 | 164 | 164 | 0.756 | 0.735 | 0.021 [-0.035, 0.076] | 0.055 [-0.090, 0.200] | -0.334 [-0.476, -0.275] | 0.533 | 8 | 7.1 |
| overall | all | highest_price | 72h | 5 | 1 | 240 | 240 | 0.746 | 0.754 | -0.008 [-0.059, 0.040] | -0.005 [-0.117, 0.121] | 0.780 [0.780, 0.780] | -0.954 | 5 | 9.9 |
| period | fit | highest_price | 72h | 5 | 1 | 116 | 116 | 0.733 | 0.734 | -0.001 [-0.062, 0.062] | -0.050 [-0.183, 0.073] | 0.104 [0.104, 0.104] | -1.000 | 2 | 14.9 |
| period | tune | highest_price | 72h | 5 | 1 | 124 | 124 | 0.758 | 0.772 | -0.014 [-0.081, 0.050] | 0.063 [-0.149, 0.283] | 1.503 [1.503, 1.503] | -0.915 | 3 | 7.5 |
| size | 3-4 | highest_price | 72h | 5 | 1 | 22 | 22 | 0.864 | 0.858 | 0.006 [-0.090, 0.088] | 0.067 [-0.004, 0.127] | 0.878 [0.878, 0.878] | -1.000 | 1 | 16.7 |
| size | 5-9 | highest_price | 72h | 5 | 1 | 93 | 93 | 0.645 | 0.669 | -0.024 [-0.107, 0.058] | -0.051 [-0.278, 0.204] | 0.779 [0.779, 0.779] | -0.952 | 4 | 13.3 |
| size | 10+ | highest_price | 72h | 5 | 1 | 125 | 125 | 0.800 | 0.798 | 0.002 [-0.056, 0.054] | 0.021 [-0.104, 0.133] |  [, ] |  | 0 | 8.5 |
| tape_only | tape_only | highest_price | 72h | 5 | 1 | 239 | 239 | 0.745 | 0.753 | -0.008 [-0.056, 0.038] | -0.006 [-0.124, 0.119] | 0.780 [0.780, 0.780] | -0.954 | 5 | 9.9 |
| overall | all | highest_price | 72h | 1 | 1 | 240 | 240 | 0.746 | 0.755 | -0.009 [-0.060, 0.036] | -0.001 [-0.114, 0.123] | 0.756 [0.756, 0.756] | -0.954 | 5 | 9.9 |
| overall | all | highest_price | 72h | 30 | 1 | 245 | 245 | 0.731 | 0.751 | -0.020 [-0.067, 0.028] | -0.003 [-0.121, 0.127] | 0.799 [0.799, 0.799] | -0.954 | 5 | 9.4 |
| stress | fee_x2_delay30 | highest_price | 72h | 30 | 2 | 245 | 245 | 0.731 | 0.751 | -0.020 [-0.067, 0.028] | -0.012 [-0.126, 0.114] | 0.799 [0.799, 0.799] | -0.954 | 5 | 9.4 |
| overall | all | highest_price | 48h | 5 | 1 | 301 | 301 | 0.744 | 0.754 | -0.010 [-0.051, 0.032] | -0.000 [-0.116, 0.106] | 0.410 [0.318, 0.492] | -0.013 | 15 | 11.1 |
| period | fit | highest_price | 48h | 5 | 1 | 157 | 157 | 0.682 | 0.718 | -0.037 [-0.099, 0.025] | 0.001 [-0.137, 0.152] | 0.250 [0.106, 0.377] | 0.557 | 11 | 12.9 |
| period | tune | highest_price | 48h | 5 | 1 | 144 | 144 | 0.812 | 0.794 | 0.019 [-0.037, 0.074] | -0.001 [-0.186, 0.189] | 0.759 [0.759, 0.759] | -1.000 | 4 | 9.1 |
| size | 3-4 | highest_price | 48h | 5 | 1 | 33 | 33 | 0.848 | 0.836 | 0.013 [-0.074, 0.090] | 0.049 [-0.076, 0.143] |  [, ] |  | 0 | 19.1 |
| size | 5-9 | highest_price | 48h | 5 | 1 | 119 | 119 | 0.639 | 0.674 | -0.035 [-0.110, 0.041] | -0.082 [-0.268, 0.112] | 0.989 [0.892, 1.090] | -0.939 | 10 | 9.3 |
| size | 10+ | highest_price | 48h | 5 | 1 | 149 | 149 | 0.805 | 0.801 | 0.005 [-0.051, 0.056] | 0.052 [-0.118, 0.228] | -0.793 [-0.793, -0.793] | 0.971 | 5 | 11.8 |
| tape_only | tape_only | highest_price | 48h | 5 | 1 | 298 | 298 | 0.742 | 0.756 | -0.015 [-0.053, 0.027] | -0.042 [-0.148, 0.063] | 0.821 [0.732, 0.920] | -0.953 | 14 | 10.8 |
| overall | all | highest_price | 48h | 1 | 1 | 300 | 300 | 0.743 | 0.754 | -0.010 [-0.053, 0.029] | -0.001 [-0.111, 0.105] | 0.400 [0.307, 0.482] | -0.013 | 15 | 11.4 |
| overall | all | highest_price | 48h | 30 | 1 | 301 | 301 | 0.741 | 0.756 | -0.015 [-0.055, 0.026] | -0.002 [-0.113, 0.115] | 0.352 [0.285, 0.450] | -0.022 | 16 | 10.6 |
| stress | fee_x2_delay30 | highest_price | 48h | 30 | 2 | 301 | 301 | 0.741 | 0.756 | -0.015 [-0.055, 0.026] | -0.012 [-0.119, 0.103] | 0.352 [0.285, 0.450] | -0.022 | 16 | 10.6 |

## Caveats

- Scheduled-reveal events (independent news time = scheduled end) are a selected subset of markets.
- Leaders are chosen from prices only; the eventual winner is never used. Events are required to be neg-risk with at least 3 outcomes comparable at selection time, and all siblings must have a close time.
- Entry uses the outcome market's own taker tape (any trade, the NO-token price converted to a YES price); the snapshot price series is used only for markets with no tape at all (then the volume cap is unknown and the full $100 stake is assumed; see the tape_only rows).
- Thin sibling tapes: entry within 6 h of the delayed time, else the trade is skipped; fills are not guaranteed at displayed size.
- The fee schedule is from a secondary source and is to be confirmed with the venue.
- The price-matched placebo (non-leader within 0.05 of the leader's entry price) exists for only a subset of trades (column 'matched trades'); where that subset is small the placebo interval is degenerate and not informative, and 'beats placebo' there is weak evidence.
- Several horizons and two rules are tested on overlapping events: results are not independent.
