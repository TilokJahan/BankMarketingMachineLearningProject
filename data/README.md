# Data

## Primary Dataset

**UCI Bank Marketing Dataset**

Source:  
https://archive.ics.uci.edu/dataset/222/bank+marketing

Do not commit the raw CSV or ZIP file unless the team explicitly decides to do so.

## Candidate File

The UCI download commonly includes:

- `bank-full.csv`

This version contains roughly 45,000 observations.

## Target

`y`

- `yes` — customer subscribed to a term deposit
- `no` — customer did not subscribe

## Feature Types

The dataset contains a mixture of:

- Numeric features such as age, balance, campaign count, and previous contact count
- Binary categorical variables such as housing loan and personal loan
- Multi-class categorical variables such as job, marital status, education, contact type, month, and previous campaign outcome

## Important Analysis Question

Before modeling, investigate whether all campaign-related variables would actually be available at the point when the model is intended to make a prediction.

Some variables may create **data leakage** or may only be known after/during a marketing call.

The team should document which variables are retained or excluded and why.
