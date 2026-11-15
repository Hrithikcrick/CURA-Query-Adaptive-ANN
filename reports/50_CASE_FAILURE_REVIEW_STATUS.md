# Fifty-case failure review status

Required human reviews: **36/50 complete**, **14 pending**.

First unfinished case: **37**, query **515**, stage **WAITING_NOTE**.
Case 37 already has the researcher's category 2. Its displayed evidence note still needs confirmation. No note acceptance was inferred.

| Field | Value |
|---|---:|
| total_cases | 50 |
| already_completed_before_this_run | 36 |
| newly_completed | 0 |
| remaining | 14 |
| human_verified | 36 |
| model_assisted | 14 |
| newly_prepared_suggestions | 13 |
| carried_forward_suggestions | 1 |

**Status: NEEDS_HUMAN_CONFIRMATION.** Model-assisted counts refer to pending suggestions, not additional completed reviews.

Completed case numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36.
Completed query IDs: 10845, 1451, 4785, 5592, 721, 8795, 904, 9824, 11088, 1198, 4946, 5125, 7105, 10034, 2316, 3085, 594, 8, 8271, 4681, 7590, 2400, 1297, 6985, 4007, 4999, 3830, 5981, 4700, 9644, 6668, 1284, 4920, 604, 7823, 10137.

The companion CSV prepares only cases 37–50: one suggestion carried forward exactly and 13 new suggestions from saved metrics and displayed document/rank evidence. It contains no new human approval. The original worksheet, checkpoint and proof log were not written.

The older assistant draft remains available for all 50 cases; its content was not reanalyzed. The separate MS MARCO/TREC taxonomy remains 50 model-assisted labels and zero human verifications.

The checkpoint was read and validated directly from the live workspace after integration. Its original human decisions and notes remain unchanged.

Source worksheet SHA256: `2936cf021358632d3117ad1c1907d21e2c975468120b635559677251492113e7`.
Checkpoint SHA256: `4475b96b231f76fcbde8efb6e707faa313cd9f1848ac308eb8f307b6d870b80c`.
Decision log SHA256: `c06722d91941185ef5a091aea4d05bd4479121a4f4af3f12d0aee32a663168c0`.

Evidence label 0 means no positive qrel was recorded in the displayed evidence; it does not establish that every passage was explicitly judged nonrelevant. Missing top-five entries do not by themselves establish absence at all ranks. No causal graph-search explanation was inferred.
