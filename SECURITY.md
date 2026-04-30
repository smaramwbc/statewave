# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | ✅                 |
| < latest | ❌ (upgrade recommended) |

## Reporting a Vulnerability

If you discover a security vulnerability in any Statewave repository, please report it responsibly.

### ⚠️ Do NOT

- Open a public GitHub issue
- Discuss the vulnerability publicly before it's fixed
- Exploit the vulnerability

### ✅ Do

1. **Email us at:** [security@statewave.ai](mailto:security@statewave.ai)

2. **Include in your report:**
   - Description of the vulnerability
   - Steps to reproduce
   - Affected repository and version
   - Potential impact assessment
   - Any suggested fixes (optional)

3. **What to expect:**
   - Acknowledgment within **48 hours**
   - Initial assessment within **5 business days**
   - Resolution timeline communicated based on severity
   - Credit in release notes (if desired)

## Severity Levels

| Severity | Response Time | Examples |
|----------|--------------|----------|
| Critical | 24 hours | RCE, data breach, auth bypass |
| High | 72 hours | Privilege escalation, XSS, SQL injection |
| Medium | 1 week | Information disclosure, CSRF |
| Low | 2 weeks | Minor issues, best practice violations |

## Security Measures

Statewave maintains security through:

- **Dependency Scanning:** Dependabot enabled on all repositories
- **Code Scanning:** GitHub CodeQL analysis on PRs
- **CI/CD Security:** All PRs require passing security checks
- **Secret Management:** Secrets via environment variables, never in code
- **Access Control:** Principle of least privilege for all systems
- **Audit Logging:** Provenance tracking for all data operations

## Responsible Disclosure

We believe in responsible disclosure and will:

- Work with you to understand and validate the issue
- Keep you informed of our progress
- Credit researchers who report valid issues (unless anonymity requested)
- Not take legal action against good-faith security research

## Scope

This policy applies to all Statewave repositories:

- `statewave` - Core backend
- `statewave-py` - Python SDK
- `statewave-ts` - TypeScript SDK
- `statewave-docs` - Documentation
- `statewave-examples` - Examples
- `statewave-demo` - Demo site
- `statewave-web` - Marketing site
- `statewave-admin` - Admin dashboard

## Contact

- **Security issues:** [security@statewave.ai](mailto:security@statewave.ai)
- **General questions:** [GitHub Discussions](https://github.com/smaramwbc/statewave/discussions)
