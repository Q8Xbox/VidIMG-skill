---
description: Restore a video faithfully (classical field handling + open models + guard, free tools only)
argument-hint: <video file> [preview]
---
Use the restore-media skill. Restore this video: $ARGUMENTS

Run `analyze_media` first and explain in English what was found (interlace/telecine, compression, noise,
faces) and the plan. If the argument includes "preview", or the video is longer than about 10 minutes,
start `restore_video` with `preview_only=true` and report the chosen model before the full run. Poll
`job_status` every 30–60 seconds until done, then report the final gates, what was delivered and the
report page path.
