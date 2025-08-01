# GitHub Actions Permissions Guide

## Required Permissions for PR Comments

When using actions that comment on pull requests, the workflow needs appropriate permissions. Add this to your workflow:

```yaml
permissions:
  pull-requests: write
  contents: read
  issues: write
```

## Token Requirements

Some actions may require the `GITHUB_TOKEN` to be explicitly passed:

```yaml
- uses: marocchino/sticky-pull-request-comment@v2
  with:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    header: coverage
    message: "Your message here"
```

## Conditional PR Comments

Only attempt to comment on actual pull requests:

```yaml
- name: Comment PR
  if: github.event_name == 'pull_request'
  uses: marocchino/sticky-pull-request-comment@v2
```

## Alternative: Use Job Summaries

For workflows that run on both PRs and pushes, consider using GitHub Job Summaries instead:

```yaml
- name: Generate Summary
  run: |
    echo "## Test Results" >> $GITHUB_STEP_SUMMARY
    echo "All tests passed!" >> $GITHUB_STEP_SUMMARY
```