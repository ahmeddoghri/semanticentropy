# Security Policy

## Supported versions

This project is pre-1.0. Security fixes land on `main`; track the latest commit.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/ahmeddoghri/semanticentropy/security/advisories/new)
or email the maintainer. Include a description of the issue and its impact,
steps to reproduce (a minimal proof-of-concept helps), and any suggested fix.

You can expect an acknowledgement within a few days. Once a fix is out you will
be credited unless you would rather stay anonymous.

## Scope notes

semanticentropy is a pure-stdlib library with no runtime dependencies and makes
no network calls. It clusters and scores text you provide, so there is very
little attack surface here. Note that this package does not call a model itself:
you sample the model yourself (five calls at nonzero temperature) and pass the
strings in. Sampling cost and any model API keys are entirely your concern, not
this package's.
