# Action: improve

You are given the full source of the current best solution (the parent) below, along with its diagnostics. Make one focused, well-motivated change along its existing direction — not a rewrite, not a grab-bag of unrelated tweaks. State clearly in `changes` exactly what you changed relative to the parent; the harness will compute the diff itself.

Use the diagnostics evidence pack (per-bucket breakdown, calibration, drift indicators) to target something the parent is specifically bad at, not a generic hyperparameter nudge. If the diagnostics don't suggest an obvious next step in this direction, say so honestly in `rationale` and propose the best available option anyway — don't fabricate false confidence.

If your last two attempts in this direction gained less than half of epsilon, you will be asked to switch direction next time — so if you can tell this direction is running out of room, say that plainly in `risks`.

If this change bundles more than one distinct idea at once (e.g. a new feature *and* a new loss *and* a new architecture choice, all in the same iteration), you have no way of knowing afterward which part actually helped and which part hurt if the result is disappointing. Prefer isolating one major change per iteration where practical, so a good or bad result is attributable to something specific rather than a tangle of simultaneous changes.
