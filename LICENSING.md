# Licensing

Statewave is **dual-licensed**:

1. **Open source** — [GNU Affero General Public License v3.0](LICENSE) (AGPLv3)
2. **Commercial** — a separate [Statewave Commercial License](COMMERCIAL-LICENSE.md)
   for proprietary, SaaS, embedded, hosted, or enterprise use.

This structure keeps Statewave open and community-driven while allowing the
project to sustain itself commercially. You can adopt Statewave under AGPLv3
with no paperwork; you only need a commercial license if your use case
conflicts with AGPL obligations or you need enterprise terms.

## Quick decision guide

| Use case                                                                              | License path                                                  |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Learning, local development, research                                                 | AGPLv3                                                        |
| Open-source application that uses Statewave and complies with AGPL                    | AGPLv3                                                        |
| Internal company use with no external SaaS exposure                                   | AGPLv3 — or commercial license if your legal team prefers     |
| Proprietary SaaS that uses Statewave as infrastructure                                | Commercial license (recommended/required to avoid AGPL §13)   |
| Hosting Statewave as a managed/SaaS service for third parties                         | Commercial license required                                   |
| Embedding Statewave in closed-source software shipped to customers                    | Commercial license required                                   |
| Early-stage startup under the qualifying threshold                                    | Startup commercial license                                    |
| Larger company needing SLA, indemnity, support, procurement, custom terms             | Enterprise commercial license                                 |

If you are unsure where you fit, contact
[licensing@statewave.ai](mailto:licensing@statewave.ai) and we will help you
pick the right path.

## What AGPLv3 requires (in plain English)

AGPLv3 is a strong copyleft license. The most important obligations:

- If you **distribute** Statewave (or a modified version), you must make the
  corresponding source available under AGPLv3.
- If you **make Statewave available to users over a network** — for example,
  as part of a SaaS or hosted product — AGPL §13 requires you to offer
  those users the corresponding source of the version you are running,
  including any modifications.
- The whole work that combines with Statewave under copyleft terms must be
  licensed under AGPLv3.

If those obligations are acceptable to you, AGPLv3 is the right choice and
no commercial license is required.

If they are **not** acceptable — for example, you ship a closed-source SaaS
product, or you embed Statewave in a proprietary distribution — use the
commercial license.

## What the commercial license offers

The Statewave Commercial License is offered in tiers
(see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) and
[docs/licensing.md](docs/licensing.md)):

- **Startup** — free or low-cost commercial license for qualifying
  early-stage companies. Removes AGPL source-disclosure obligations for
  covered use. Does not include the right to host Statewave as a competing
  service.
- **Growth** — paid commercial license for production SaaS / proprietary
  product use above the startup threshold.
- **Enterprise** — custom commercial agreement with SLA, indemnity, support,
  optional managed hosting, and procurement-friendly terms.

A commercial license is the right choice when you want closed-source, SaaS,
embedded, or hosted use without AGPL obligations.

## Commercial protection (what you may not do)

The dual-license model is designed to be adoption-friendly while protecting
the project. The following uses require either full AGPLv3 compliance **and**
compliance with the [trademark policy](TRADEMARKS.md), or a commercial
agreement:

- Offering Statewave (or a substantially similar derivative) as a hosted,
  managed, or SaaS service to third parties.
- Distributing closed-source software that links, embeds, or otherwise
  combines with Statewave in a way that triggers AGPL copyleft.
- Using the Statewave name, logo, or branding to imply official hosting,
  endorsement, or affiliation. See [TRADEMARKS.md](TRADEMARKS.md).

Running Statewave for internal use, building open-source projects on top of
it, contributing back, and sharing modifications under AGPLv3 are all
explicitly welcome and require no agreement.

## How to get a commercial license

Email [licensing@statewave.ai](mailto:licensing@statewave.ai) with:

- Company name and a short description of intended use
- Whether the use is internal, embedded, SaaS, or hosted-for-third-parties
- Approximate scale (users, environments, revenue band)
- Any required contractual terms (SLA, indemnity, support, procurement)

We will respond with the appropriate tier and a draft agreement.

## Contributing under dual licensing

External contributions must be compatible with this dual-licensing model.
By submitting a contribution you agree that it can be distributed under
AGPLv3 **and** under the Statewave Commercial License. We may ask
contributors to sign a CLA or DCO before contributions are accepted. See
[CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Disclaimer

This repository describes Statewave's licensing model and is not legal
advice. Consult qualified counsel before adopting Statewave in a
commercial product. If you have questions about how AGPLv3 applies to
your specific situation, please consult your own attorney.
