---
description: Restore anything automatically (photos, videos, series folders, or a mix): survey, decide, run, one report
argument-hint: <files and/or folders>
---
Use the restore-media skill and its automatic flow. Restore: $ARGUMENTS

1. `survey_media(paths)` (background job); poll `job_status` until done. Its result lists every item (the photo
   batch, each single video, each numbered series) with a decision already filled in, and `contact_sheets`.
2. Open every contact sheet image and every video's `closeups` image and look. Judge each `findings` entry on
   the close-ups (`crop_borders` when only black lies outside the red outline, `remove_repeats` when the frames
   marked COPY are padding, not held drawings, `filters: ["deinterlace_scenes"]` when the comb close-ups show
   teeth on moving edges) and compose the clean-up the line close-ups call for (`filters` with `deblock` or
   `dehalo` at the mildest strength that removes the defect). When a look must settle between recipes, compare them with
   `try_recipes` first. For each video and series item decide drawn (anime) or real footage,
   and whether another model fits better (`guide` shows the pick for either reading; realistic drawn faces:
   cugan-pro). Send all corrections in one `decide_plan(plan, {...})` call, confirming every video item and giving
   a short `reason`.
3. Tell the person in a few lines what was found and what you decided and why. Do not ask them to choose; ask only
   about opt-in steps they did not mention when the survey shows they apply (blur removal for soft photos,
   masking tape noise).
4. `run_plan(plan)` (background job); poll `job_status` every 30–60 s. While it runs the result shows finished items.
5. Open the before/after sheets of every delivered video (`result.sheets`) and record what you see with
   `review_results` (good / acceptable / redo, with notes).
6. Report from the result in English: per item, delivered or original kept and why, any guard warnings, your review, and the
   path of the combined report (`report`, index.html). Ask the person to look at the faces in the reports.
