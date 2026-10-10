# Contributing / Team Workflow

This is an initial suggestion and can be changed after the team discussion.

## Recommended workflow

1. Pull the latest `main` branch before starting work.
2. Create a branch for each task.
3. Make focused commits with clear messages.
4. Push the branch to GitHub.
5. Open a pull request.
6. Have at least one teammate review before merging when practical.

## Example branches

```bash
git checkout -b eda-initial
git checkout -b preprocessing-pipeline
git checkout -b baseline-models
```

## Suggested first parallel tasks

### Team Member 1
Initial EDA and data-quality / leakage investigation.

### Team Member 2
Initial preprocessing pipeline and train/test strategy.

### Team Member 3
Baseline model and reusable evaluation framework.

The team should regroup before locking in the final feature set and full model comparison.
