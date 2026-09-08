# SIGF advisor review, September 4, 2026

The new labeled review is at https://clinician-evaluation.vercel.app/sigf . It is separate from the older blinded GRAPE/PAPILA pilot at `/`.

The frozen set contains 40 distinct test patients, one eye per patient, three actual follow-ups per eye, and 280 image files. The private source manifest, model provenance, and image hashes are under `outputs/review/sigf_advisor_40_20260904/`. Meeting artifacts are under `outputs/advisor_meeting_20260904/deliverables/` and copied to Downloads.

## Selection and generation

Eligible eyes have at least three distinct observations 0.25–5 years after a baseline. The script uses the earliest eligible baseline, selects distinct visits nearest 1, 3, and 5 years, sorts them chronologically, ranks eligible eyes by their previously measured mean test SSIM, and takes one eye per patient until reaching 40. This is a deliberately curated showcase, with selection on test performance; it is not an unbiased estimate or a new independent test. Stable eyes are included.

The checkpoint is `sigf_multitask_diffusion_v3_20260816/best.pt`. Inference uses only baseline-derived inputs and the actual observed intervals, two sampled trajectories, 15 diffusion steps, and the frozen validation change scale of 0.5680599113752357. The displayed AI image is the mean of the trajectories. Generated images never become baseline observations.

Review photographs are original 256-pixel optic-disc crops with consistent left-eye mirroring. There is no registration or color matching in the review. There are no automatic contour overlays. Full-test metrics use their original evaluation pipeline and should not be represented as newly computed scores on these displayed crops.

## App and storage

New routes: `/sigf`, `/sigf/results`, `/api/sigf`, and login/image/responses/export subroutes. Signed HttpOnly cookies authenticate reviewer codes. Private Vercel Blob stores both images and responses under `sigf/sigf-advisor-curated-20260904-v1/`. The browser sees anonymous case IDs, actual elapsed intervals, and laterality; source patient identifiers and paths remain local.

Environment keys: `SIGF_REVIEW_PASSCODE`, `SIGF_ADMIN_EXPORT_TOKEN`, and the existing `BLOB_READ_WRITE_TOKEN`. The new export key is separate from the original pilot's administrator key. The original root pilot and storage namespace remain intact. Never include passcodes or `.env` contents in study exports, slides, or source control.

A case requires all three follow-up ratings before saving. The API validates response options and assigns elapsed intervals from the server-side manifest. Edits replace the same reviewer/case record. The CSV contains one row per follow-up, escapes spreadsheet formula prefixes, and excludes reviewer codes beginning `SYSTEM-TEST`. Unsubmitted drafts stay in the browser; completed series resume from the server.

## Interpretation

Collect similarity (1–5 or unable), observed photographic change (none/possible/definite/unable), and relative AI change (too little/similar/too much/unable), with optional comments. Ratings are unblinded formative feedback from one advisor. Confirm the applicable institutional determination before formal research use. Do not claim inter-reader reliability, diagnostic validity, or population accuracy from this review.

Validation covered all 280 private images, chronology, distinct patients, sign-in and access denial, invalid request rejection, save/resume/update behavior, and protected CSV export. Synthetic verification responses are excluded from the CSV and removed after verification.

## Access update

At the user’s request, the review opens without sign-in. The manifest endpoint automatically issues a signed anonymous browser session, preserving existing valid reviewer sessions. Photos require only this automatically created session. Response records remain separate by browser session; the owner export still requires its separate results passcode. The old login endpoint is removed. SIGF_REVIEW_PASSCODE now serves only as the server-side signing secret.


## Final-follow-up grading update

The review now opens at visit index 2 and requires one rating per patient, for that final available follow-up. The first two visits remain optional reference views without grading controls. Actual elapsed intervals remain unchanged, so the final visit is not relabeled as 4.93 years for every patient. The page retains the AI display contrast control and the Original option.

New submissions contain only the index-2 rating. Saving merges that rating into any existing patient response, preserving older index-0 and index-1 answers in storage. The CSV export now includes only index 2, with its actual elapsed interval. Existing complete final ratings count toward progress and resume correctly.
