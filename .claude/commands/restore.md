---
description: Restore anything automatically (photos, videos, series folders, or a mix): survey, decide, run, one report
argument-hint: <files and/or folders>
---
Use the restore-media skill and its automatic flow. Restore: $ARGUMENTS

1. `survey_media(paths)` (background job); poll `job_status` until done. Its result lists every item (the photo
   batch, each single video, each numbered series) with a decision already filled in, and `contact_sheets`.
2. Open every contact sheet image and look. For each video and series item decide drawn (anime) or real footage,
   and whether another model fits better (`guide` shows the pick for either reading; realistic drawn faces:
   cugan-pro). Send all corrections in one `decide_plan(plan, {...})` call, confirming every video item and giving
   a short `reason`.
3. Tell the person in a few lines what was found and what you decided and why. Do not ask them to choose; ask only
   about opt-in steps they did not mention when the survey shows they apply (blur removal for soft photos,
   masking tape noise).
4. `run_plan(plan)` (background job); poll `job_status` every 30–60 s. While it runs the result shows finished items.
5. Report from the result in English: per item, delivered or original kept and why, any guard warnings, and the
   path of the combined report (`report`, index.html). Ask the person to look at the faces in the reports.
