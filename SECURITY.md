# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | ✅                 |
| < latest | ❌ (upgrade recommended) |

## Reporting a Vulnerability

If you discover a security vulnerability in any Statewave repository, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email: **security@statewave.ai**
3. Include: description, reproduction steps, affected repo/version, potential impact.

We will acknowledge within 48 hours and aim to resolve critical issues within 7 days.

## Security Measures

- Dependabot is enabled on all repositories for automated dependency updates.
- GitHub code scanning (CodeQL) is enabled for backend and frontend repos.
- All PRs require passing CI before merge.
- Secrets are managed via environment variables, never committed to source.
