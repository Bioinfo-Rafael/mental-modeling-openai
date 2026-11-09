# MHI v1.3 match data (LAG 2v2)

Three committed LAG 2v2 matches, shared by the core (Translator, judge, tests) and, read-only, by
`workspace/13_LAG`–`17_commentary_LAG`. The files are byte-identical copies moved here from
`workspace/13_LAG/data/` during the core/workspace split; do not edit them.

| Path | Content | Readers |
|---|---|---|
| `processed/{mhi_v1.3_ic100601_win,ours_seed100100_ic100601_loss,ours_seed100100_ic100606_win}.csv` | 64-column episode CSVs, one row per (step, object) | `translate.episode.load_episode`, the example scenario, the judge (FUTURE, trajectory), tests |
| `processed/manifest.json` | schema, SHA-256 of every CSV, provenance | `translate.episode.load_episode` verifies each CSV against it (keep CSV and manifest together) |
| `processed/README.md` | column definitions and loading rules | humans |
| `action_videos/*_with_actions.mp4` | recording + A-team action plots, 20 fps, 1920x1080, frame = step | human raters (the judge reads no video yet) |

Provenance: the raw ACMI, action CSVs and recordings stay in `workspace/13_LAG/data/raw/`.
`workspace/13_LAG/run.py` regenerates `processed/` from them (`--check` compares without writing),
and `workspace/13_LAG/data/viz/render_action_videos.py` rendered `action_videos/`. The example path in
`processed/README.md` still names the old `workspace/13_LAG/data/processed/` location, because the file
is kept byte-identical.
