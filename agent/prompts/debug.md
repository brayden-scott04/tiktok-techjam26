# Action: debug

The solution below failed. You are given the exact error and, where available, a traceback, stdout tail, peak memory, and elapsed time. Fix the specific problem — do not use this as an opportunity to redesign the approach; that belongs in a later `improve` step once this version actually runs.

Common causes worth checking first:
- A numpy 2.x removed name (`np.float_`, `np.NaN`, `np.in1d`, `np.alltrue`, etc.)
- A shape mismatch between `ctx.splits[...]` arrays of different lengths
- Exceeding `ctx.max_seconds` or the `ctx.eval_valid` call cap (200/run) — reduce evaluation frequency or training cost, don't just catch the exception
- A banned import or file-access attempt — remove it; you don't need file I/O, everything you need is on `ctx`

Set `hypothesis` to a one-line restatement of the original idea (unchanged), and describe the fix itself in `changes`.
