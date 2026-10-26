# Data

The data files are not included in this repository because of their size.

**Source:** [Smoking and Drinking Dataset with body signal](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset) on Kaggle. It holds about 991k health-screening records from the National Health Insurance Service in Korea, with personal information removed.

The notebook expects three preprocessed CSV files in this folder:

| File | Contents |
|---|---|
| `scaled_smoking_drinking_dataset.csv` | All records |
| `scaled_smoking_drinking_dataset_50_and_under.csv` | Records with age ≤ 50 |
| `scaled_smoking_drinking_dataset_above_50.csv` | Records with age > 50 |

Preprocessing applied to produce them:
- Continuous measurements were standardized (z-scores).
- `sex` was encoded as 0/1.
- `DRK_YN` was encoded as 0/1.
- Smoking status `SMK_stat_type_cd` was binarized to 0/1.
- Hearing and urine-protein codes were kept as integers.

The preprocessing script was not part of the original submission.
