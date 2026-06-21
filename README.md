# Mini-project-2

Simple spam detector — minimal, no external dependencies.

Usage:

- Run interactively:

```bash
python3 spam_detector.py
```

- Classify a single message from the command line:

```bash
python3 spam_detector.py "Congratulations, you won free cash!"
```

The detector uses a tiny rule-based heuristic (keyword matches, URL,
exclamation marks). It is intentionally simple and easy to read.