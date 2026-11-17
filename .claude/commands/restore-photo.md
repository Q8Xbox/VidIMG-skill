---
description: Restore photos faithfully (files or folders) with the fidelity guard
argument-hint: <photo files or folders>
---
Use the restore-media skill. Restore these photos: $ARGUMENTS

Analyse a sample with `analyze_media`, summarise the findings and plan in Arabic, start `restore_photos`
(pass `deblur=true` only if the person asked to remove blur),
poll `job_status` until it finishes, then report per photo (delivered or kept original, and why) with the
path of the review page. Do not change the approved policy (free tools only, no generative models, no
colour/lighting changes).
