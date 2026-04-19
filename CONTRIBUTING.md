# Contributing

Thanks for contributing to this project.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development workflow

1. Create a branch from `main`.
2. Make your changes in focused commits.
3. Run tests before opening a pull request.
4. Keep changes scoped and easy to review.

## Local checks

```bash
python3 -m unittest discover -s tests
python3 train_model.py
```

## Pull request guidance

- Describe the problem clearly.
- Summarize the solution.
- Mention any behavior changes.
- Include screenshots for UI changes when possible.

## Code style

- Keep Python code readable and simple.
- Prefer clear names over short names.
- Add comments only when they help explain logic.
