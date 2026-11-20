# Meta-Harness integration

Use the public launcher and setup instructions in the [release README](../../README.md).
This directory holds the Meta-Harness search loop (`mh_loop.py`), the RSI-top breadth
adapter (`mh_breadth.py`), and the fork entry point (`run_repeat_flat_breadth.py`).

Unlike OpenEvolve and Shinka, which start a 32-direction cohort at once, this loop runs at
most 8 Pi sessions at a time, started 3 s apart, and queues the rest of the cohort; Cloudflare
Workers AI rejects larger bursts with 429. A session that ends on 429/408 without a candidate is
rerun after 30–480 s backoff (up to 5 times) and recorded once; one that still fails is
`rate_limited` and is not reported to the planner as a failed direction.
