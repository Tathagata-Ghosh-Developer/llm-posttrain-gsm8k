| model | greedy pass@1 | pass@1 (T=0.7) | pass@4 | pass@8 | format rate |
|---|---|---|---|---|---|
| base | 34.3* | n/a | n/a | n/a | 0.4 |
| base_4shot | 35.0 | 26.5 | 51.1 | 63.5 | 94.1 |
| sft | 34.6 | 27.2 | 51.9 | 64.3 | 97.3 |
| dpo | 29.2 | 21.5 | 43.6 | 56.4 | 82.9 |
| grpo | 44.7 | 40.5 | 62.4 | 71.2 | 98.2 |

* Re-scored from the saved greedy outputs after the answer-extraction fix (`rescored_greedy.csv`; the logged value was 5.5). The base model's sampled completions were not saved, so its pass@k is not reported; the logged values (7.7 / 23.8 / 36.9) came from the old extractor and are not valid.
