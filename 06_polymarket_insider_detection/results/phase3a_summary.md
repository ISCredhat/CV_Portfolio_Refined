# Phase 3A: wallet tests (aggregate only; statistical flags, no addresses)

- wallets_with_resolved_positions: 21156
- wallets_ge5_positions: 205
- raw_p_pb_lt_alpha: 1
- bh_flag_pb: 0
- raw_p_boot_lt_alpha: 5
- bh_flag_boot: 0
- bh_flag_both: 0
- expected_false_raw: 10.25

Phase 3b: calibration fit and repeat test use calib_eligible positions only (independent-news cutoff; jump_start cutoffs are outcome-selected and excluded); see results/phase3b_calibration.csv for the entry-price decile table before/after.
- calib_eligible resolved positions in fit: 25859; wallets: 21156
- info_type counts in fit (min per-type training size 500, else pooled fallback): scheduled_reveal=24519, deadline=1340
- best wallet (aggregate stats only): {'n_pos': 15, 'wins': 14, 'expected': 11.252192788301208, 'p_pb': 0.02280233866739484, 'q_pb': 0.9958970977677623, 'p_boot': 0.020795840831833633, 'q_boot': 0.9004300834748304, 'excess_pnl': 355.5186501812472}

p_pb histogram over eligible wallets (10 bins on [0,1]):

- 11
- 15
- 19
- 12
- 29
- 27
- 20
- 31
- 20
- 21

Caveat (phase 3b): jump_start cutoffs are outcome-informed (a jump is detected toward the winner), so cheap winners / expensive losers are over-represented in that subset (win rate ~0.97-0.99 at prices 0.1-0.3, ~0.03 at 0.8+); the independent-news-only subset is close to the diagonal. See the after_* rows in results/phase3b_calibration.csv.

Power is low: only the 399 pilot markets plus extras; most wallets have 1-2 positions.
