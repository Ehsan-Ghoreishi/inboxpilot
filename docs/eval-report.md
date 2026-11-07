# Classification eval report

| Metric | dev | test | dev + test |
|---|---|---|---|
| Results file timestamp | 2026-11-07T22:07:25.706126+00:00 | 2026-11-07T22:17:01.290184+00:00 | 2026-11-07T22:07:25.706126+00:00, 2026-11-07T22:17:01.290184+00:00 |
| Prompt version | analyze_v2 | analyze_v2 | analyze_v2 |
| Model | gpt-4o | gpt-4o | gpt-4o |
| Dataset SHA-256 | 2704a01c11db9624e88f53b4bff751f883b27a323ea4bd5268bc96b800cefab0 | 2704a01c11db9624e88f53b4bff751f883b27a323ea4bd5268bc96b800cefab0 | 2704a01c11db9624e88f53b4bff751f883b27a323ea4bd5268bc96b800cefab0 |
| Items | 52 | 52 | 104 |
| Errors | 0 | 0 | 0 |
| Accuracy | 0.981 | 0.981 | 0.981 |
| Accuracy 95% CI (Wilson) | 0.899 to 0.997 | 0.899 to 0.997 | 0.933 to 0.995 |
| interview_invite support | 15 | 16 | 31 |
| interview_invite recall | 0.933 | 1.000 | 0.968 |
| interview_invite recall 95% CI (Wilson) | 0.702 to 0.988 | 0.806 to 1.000 | 0.838 to 0.994 |
| assessment_deadline support | 9 | 8 | 17 |
| assessment_deadline recall | 1.000 | 1.000 | 1.000 |
| assessment_deadline recall 95% CI (Wilson) | 0.701 to 1.000 | 0.676 to 1.000 | 0.816 to 1.000 |
| Mean input tokens per email | 3137.0 | 3133.7 | 3135.3 |
| Mean output tokens per email | 95.0 | 96.3 | 95.6 |
| Cost per email (USD) | n/a | $0.008797 | n/a |

## Recall of the costly classes

| Class | Recall | Missed | Support | Missed as |
|---|---:|---:|---:|---|
| interview_invite | 0.968 | 1 | 31 | recruiter_outreach: 1 |
| assessment_deadline | 1.000 | 0 | 17 | — |

## Per-class precision, recall and F1

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| recruiter_outreach | 0.889 | 1.000 | 0.941 | 16 |
| interview_invite | 1.000 | 0.968 | 0.984 | 31 |
| assessment_deadline | 1.000 | 1.000 | 1.000 | 17 |
| rejection | 1.000 | 1.000 | 1.000 | 18 |
| other | 1.000 | 0.955 | 0.977 | 22 |

## Confusion matrix (rows: actual, columns: predicted)

| Actual | recruiter_outreach | interview_invite | assessment_deadline | rejection | other | (error) |
|---|---:|---:|---:|---:|---:|---:|
| recruiter_outreach | 16 | 0 | 0 | 0 | 0 | 0 |
| interview_invite | 1 | 30 | 0 | 0 | 0 | 0 |
| assessment_deadline | 0 | 0 | 17 | 0 | 0 | 0 |
| rejection | 0 | 0 | 0 | 18 | 0 | 0 |
| other | 1 | 0 | 0 | 0 | 21 | 0 |

## Accuracy by language

| Slice | N | Accuracy |
|---|---:|---:|
| de | 26 | 1.000 |
| en | 72 | 0.972 |
| mixed | 6 | 1.000 |

## Accuracy by difficulty

| Slice | N | Accuracy |
|---|---:|---:|
| easy | 45 | 1.000 |
| hard | 30 | 1.000 |
| medium | 29 | 0.931 |

## Accuracy by id slice

| Slice | N | Accuracy |
|---|---:|---:|
| adv | 5 | 1.000 |
| amb | 10 | 0.900 |
| de | 14 | 1.000 |
| plain | 75 | 0.987 |

## Error analysis

| Id | Split | Actual | Predicted | Confidence | Dataset note |
|---|---|---|---|---:|---|
| hard-10 | dev | interview_invite | recruiter_outreach | 0.950 | asks for availability for a screen; concrete interview, no time |
| oth-09-amb | test | other | recruiter_outreach | 0.850 | — |

## Limitations

| Limitation | What it means | Evidence from these results |
|---|---|---|
| Synthetic data | Emails are written for the eval, not sampled from real inboxes (real email content is never committed), so real phrasing, noise and formatting are under-represented. | By construction |
| Small n | Few items per class, so every rate has a wide interval; a difference smaller than the interval is not evidence of a difference. | Items: dev 52, test 52, dev + test 104; accuracy 95% CI 0.933 to 0.995 |
| Saturation | Accuracy near the ceiling cannot rank prompt versions: remaining misses are few and may be label judgment calls, not model errors. | Accuracy: dev 0.981, test 0.981, dev + test 0.981 |
| Test split used once | The test split only stays an unbiased estimate while it is run once, after tuning; each extra run or follow-up change turns it into a second dev set. | Test result files found: 1 |
| Time and duration extraction n too small | Only a handful of items carry time labels, so those accuracies are anecdotal and are not reported with an interval. | Items with a time label: 10; with a duration label: 7 |
| Same-model family only | Only one model family has been evaluated, so results say nothing about other models, and a shared family may share blind spots with whoever wrote the labels. | Models: gpt-4o |
