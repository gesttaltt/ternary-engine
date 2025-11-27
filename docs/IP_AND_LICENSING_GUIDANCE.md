# Intellectual Property and Licensing Guidance

**Doc-Type:** Legal/Strategic Reference · Version 1.0 · Updated 2025-11-27

**IMPORTANT:** This document provides technical analysis and strategic guidance. It is NOT legal advice. Jonathan Verdun should consult with qualified intellectual property attorneys in the EU and/or US before making final licensing decisions.

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Technology Classification](#2-technology-classification)
3. [Licensing Options Analysis](#3-licensing-options-analysis)
4. [Regulatory Considerations](#4-regulatory-considerations)
5. [IP Protection Mechanisms](#5-ip-protection-mechanisms)
6. [Recommended Actions](#6-recommended-actions)
7. [Funding and Sustainability](#7-funding-and-sustainability)

---

## 1. Current State Assessment

### 1.1 Current License

**License:** Apache License 2.0
**Copyright Holder:** Jonathan Verdun (Ternary Engine Project)
**Year:** 2025

**Files:**
- `LICENSE` - Full Apache 2.0 text
- `NOTICE` - Attribution notice with innovation claims
- `CONTRIBUTING.md` - CLA implicit (contributions under Apache 2.0)

### 1.2 What Apache 2.0 Currently Permits

| Right | Granted | Notes |
|-------|---------|-------|
| Commercial use | Yes | Anyone can use in commercial products |
| Modification | Yes | Anyone can modify without sharing changes |
| Distribution | Yes | Anyone can redistribute |
| Patent grant | Yes | Contributors grant patent rights |
| Sublicensing | Yes | Licensees can sublicense |
| Private use | Yes | No disclosure requirements |

### 1.3 What Apache 2.0 Requires

| Requirement | Description |
|-------------|-------------|
| License inclusion | Must include copy of Apache 2.0 |
| Copyright notice | Must retain copyright notices |
| State changes | Must state significant changes made |
| NOTICE preservation | Must include NOTICE file contents |

### 1.4 Current IP Protection

**OpenTimestamps System:** Implemented
- Location: `opentimestamps/`
- Hash algorithm: SHA512
- Purpose: Establish prior art dates via Bitcoin blockchain
- Status: Active, timestamps generated

**Prior Art Established:**
- Core SIMD operations
- Dense243 encoding
- Fusion operations
- TritNet architecture

---

## 2. Technology Classification

### 2.1 Potentially Disruptive Capabilities

| Capability | Impact | Dual-Use Risk |
|------------|--------|---------------|
| 8× memory reduction for AI models | High | Medium |
| 35 Gops/s ternary throughput | Medium | Low |
| TritNet (neural arithmetic) | Very High | High |
| Edge AI deployment | High | Medium |
| Custom hardware accelerator designs | Very High | High |

### 2.2 Dual-Use Considerations

**Civilian Applications:**
- Edge AI (smartphones, IoT)
- Model compression for deployment
- Energy-efficient inference
- Academic research

**Sensitive Applications:**
- Autonomous systems (drones, vehicles)
- Surveillance systems
- Military AI applications
- Cryptographic implementations

### 2.3 Export Control Relevance

**US (EAR/ITAR):**
- Software for AI acceleration may fall under EAR Category 4 (Computers)
- Custom hardware designs may require BIS review
- Open source publication may qualify for TSR (Technology and Software Unrestricted) exception

**EU (Dual-Use Regulation 2021/821):**
- AI-enabling technologies under increased scrutiny
- May require export authorization for certain destinations
- Academic exception may apply for research

---

## 3. Licensing Options Analysis

### 3.1 Option A: Maintain Apache 2.0 (Current)

**Pros:**
- Maximum adoption and community growth
- Industry-friendly (enterprise acceptance)
- Compatible with most other licenses
- Simplifies contribution process

**Cons:**
- No revenue from commercial use
- Competitors can use without contribution
- No control over military/surveillance use
- Patent grant may limit future monetization

**Best for:** Pure open source, academic impact, community building

### 3.2 Option B: Dual Licensing (Open + Commercial)

**Structure:**
- Open source under AGPL-3.0 (copyleft)
- Commercial license for proprietary use

**Pros:**
- Revenue from commercial deployments
- Open source community still benefits
- Maintains academic accessibility
- Control over proprietary usage

**Cons:**
- Complex to administer
- May reduce enterprise adoption
- Requires legal infrastructure
- Contributors need CLA

**Examples:** MongoDB, Qt, MySQL

**Best for:** Sustainable funding while maintaining openness

### 3.3 Option C: Source-Available with Restrictions

**Structure:**
- Visible source code
- Restrictions on commercial use and/or specific applications
- Custom license (e.g., Elastic License 2.0, SSPL)

**Pros:**
- Full control over commercialization
- Can exclude specific use cases (military, surveillance)
- Transparency maintained
- Revenue potential

**Cons:**
- Not OSI-approved "open source"
- Reduces community adoption
- May face legal challenges
- Limits academic use

**Examples:** Elastic, MongoDB (SSPL)

**Best for:** Maximum control with transparency

### 3.4 Option D: Ethical/Responsible AI License

**Structure:**
- Open source with ethical use restrictions
- Prohibits harmful applications
- Requires impact assessment for certain uses

**Pros:**
- Aligns with responsible AI principles
- Prevents misuse while enabling benefit
- Demonstrates ethical leadership
- May attract ethical investors/partners

**Cons:**
- Enforcement difficult
- "Harm" definitions subjective
- May not be OSI-compliant
- Reduces adoption

**Examples:** Hippocratic License, AI Pact License

**Best for:** Values-driven development

### 3.5 Option E: Delayed Open Source (Time-Bombed)

**Structure:**
- Commercial license initially
- Becomes open source after N years (e.g., 3-5 years)
- Or after funding threshold reached

**Pros:**
- Early revenue window
- Eventual open source benefit
- Balances commercial and community interests
- Clear timeline for all parties

**Cons:**
- Complex to implement
- May discourage early adoption
- Requires version management
- Trust issues

**Examples:** MariaDB BSL, CockroachDB

**Best for:** Balancing early monetization with eventual openness

### 3.6 Comparison Matrix

| Criteria | Apache 2.0 | Dual License | Source-Available | Ethical | Delayed |
|----------|------------|--------------|------------------|---------|---------|
| Revenue potential | Low | High | High | Medium | High |
| Community adoption | High | Medium | Low | Medium | Medium |
| Academic use | Easy | Easy | Restricted | Easy | Depends |
| Military use control | None | Some | Yes | Yes | Temporary |
| Legal complexity | Low | High | Medium | High | High |
| Enterprise acceptance | High | Medium | Low | Low | Medium |

---

## 4. Regulatory Considerations

### 4.1 EU AI Act (2024)

**Relevance:** The EU AI Act classifies AI systems by risk level.

**Potential Classifications:**
- General-purpose AI model → May require transparency obligations
- High-risk applications → Conformity assessment required
- Prohibited uses → Cannot be enabled

**Recommendations:**
- Document intended use cases clearly
- Exclude prohibited applications in license
- Prepare conformity documentation if needed

### 4.2 US Export Controls

**Bureau of Industry and Security (BIS):**
- Review EAR Category 4 (Computers) and 5 (Telecommunications)
- AI/ML software may require classification
- Open source publication may qualify for exemption

**Actions:**
- Consider voluntary classification request
- Document public availability (if open source)
- Track export destinations if distributing binaries

### 4.3 Patent Considerations

**Current State:**
- No patents filed (assumption)
- Apache 2.0 includes patent grant
- OpenTimestamps establishes prior art

**Options:**
1. **Defensive patents:** File to prevent others from patenting
2. **No patents + prior art:** Rely on timestamps for defense
3. **Patent pool:** Join or create defensive pool
4. **Commercial patents:** File for licensing revenue

**Recommendation:** Consult patent attorney before publishing under current license if patents desired.

---

## 5. IP Protection Mechanisms

### 5.1 Current Protections

| Mechanism | Status | Coverage |
|-----------|--------|----------|
| Copyright | Automatic | All source code |
| Trade secret | Partial | Non-public portions only |
| OpenTimestamps | Active | Prior art for inventions |
| Trademark | Not filed | "Ternary Engine" unregistered |

### 5.2 Recommended Additional Protections

**Trademark Registration:**
- Register "Ternary Engine" in EU/US
- Protects brand even if code is open
- Prevents impersonation/confusion
- Estimated cost: €1,500-3,000 (EU), $2,000-4,000 (US)

**Patent Consideration:**
- Key innovations to consider:
  - Dense243 encoding method
  - TritNet architecture
  - Canonical indexing for SIMD
  - Fusion operation patterns
- Decision needed before broader publication
- Provisional patent (US): ~$1,500-3,000
- Full patent: $10,000-30,000+

**Copyright Registration (US):**
- Provides statutory damages in infringement cases
- Cost: ~$65 per work
- Recommended for key files

### 5.3 OpenTimestamps Enhancement

**Current coverage:** Source files, configurations

**Recommended additions:**
- Architecture documentation
- Algorithm descriptions
- Performance data
- TritNet specifications

**Process:**
```bash
# Create comprehensive timestamp
python opentimestamps/timestamp_create.py --include-docs --include-benchmarks
```

---

## 6. Recommended Actions

### 6.1 Immediate (Before Further Publication)

| Priority | Action | Timeline | Cost |
|----------|--------|----------|------|
| Critical | Consult IP attorney | This week | €500-2,000 |
| Critical | Decision on license change | Before next release | - |
| High | Create comprehensive timestamp | Today | Free |
| High | Document all innovations | This week | Time only |
| Medium | Trademark search | This week | €100-500 |

### 6.2 Short-Term (1-3 Months)

| Priority | Action | Notes |
|----------|--------|-------|
| High | File trademark application | EU/US |
| High | Provisional patent (if pursuing) | US priority date |
| Medium | Establish contributor agreement | If changing license |
| Medium | Create governance structure | Foundation or company |

### 6.3 Long-Term Considerations

| Consideration | Options |
|---------------|---------|
| Legal entity | Foundation (non-profit) vs Company (for-profit) |
| Jurisdiction | EU (Spain?) vs US (Delaware?) |
| Funding model | Grants, commercial licenses, support contracts |
| Governance | Solo, advisory board, community |

---

## 7. Funding and Sustainability

### 7.1 Funding Options by License

| License | Compatible Funding |
|---------|-------------------|
| Apache 2.0 | Grants, donations, support contracts |
| Dual License | Commercial licenses, SaaS, support |
| Source-Available | Commercial licenses, SaaS |
| Ethical License | Grants, ethical investors, donations |

### 7.2 Potential Funding Sources

**Public Funding (EU):**
- Horizon Europe (AI/HPC calls)
- Digital Europe Programme
- National innovation agencies
- EIC Accelerator (for commercialization)

**Public Funding (US):**
- NSF SBIR/STTR
- DARPA (if applicable scope)
- DOE (energy efficiency angle)

**Private Funding:**
- AI/ML focused VCs
- Strategic investors (chip companies)
- Corporate R&D partnerships
- Crowdfunding (Kickstarter, Open Collective)

### 7.3 Sustainability Models

**Model A: Foundation (Non-Profit)**
- Focus: Community benefit, academic impact
- Funding: Grants, donations, corporate sponsors
- License: Open source (Apache 2.0, GPL)
- Examples: Linux Foundation, Apache Foundation

**Model B: Company (For-Profit)**
- Focus: Commercial applications, products
- Funding: Investment, revenue, licenses
- License: Dual or proprietary
- Examples: Databricks, Elastic

**Model C: Hybrid**
- Foundation holds IP, company commercializes
- Focus: Balance community and commercial
- Funding: Mixed
- Examples: Mozilla, Docker

---

## 8. Key Decision Points for Jonathan Verdun

### 8.1 Fundamental Questions

1. **Primary goal:** Academic impact vs commercial success vs both?
2. **Control:** How much control over use cases is needed?
3. **Revenue:** Is self-sustaining funding required?
4. **Timeline:** How urgent is monetization?
5. **Values:** Are there uses you want to explicitly prevent?

### 8.2 License Change Considerations

**If changing from Apache 2.0:**
- Cannot retroactively change for existing releases
- New version can have new license
- Must handle existing contributors (may need consent)
- Community reaction should be anticipated

**If adding restrictions:**
- Document reasoning clearly
- Consider grandfather clauses for existing users
- Provide clear migration path

### 8.3 Regulatory Engagement

**Recommended approach:**
1. Proactive engagement with regulators
2. Document responsible AI practices
3. Participate in standards bodies
4. Build relationships before required

---

## 9. Resources and Contacts

### 9.1 Legal Resources

**EU:**
- EUIPO (trademarks): https://euipo.europa.eu
- EPO (patents): https://www.epo.org
- Spain Patent Office: https://www.oepm.es

**US:**
- USPTO: https://www.uspto.gov
- Copyright Office: https://www.copyright.gov

### 9.2 Open Source Legal

- Software Freedom Law Center: https://softwarefreedom.org
- Open Source Initiative: https://opensource.org
- Creative Commons (for docs): https://creativecommons.org

### 9.3 Funding

- EU Funding Portal: https://ec.europa.eu/info/funding-tenders
- Open Collective: https://opencollective.com
- GitHub Sponsors: https://github.com/sponsors

---

## Summary

The Ternary Engine represents significant innovation with potential for both beneficial and sensitive applications. The current Apache 2.0 license maximizes openness but provides no revenue stream or use-case control.

**Key recommendations:**
1. **Consult qualified IP attorney immediately** before further publication
2. **Create comprehensive timestamp** of all innovations today
3. **Decide on licensing strategy** based on goals and values
4. **Consider trademark registration** regardless of license choice
5. **Evaluate patent strategy** before broader publication
6. **Engage with relevant regulators** proactively

The author, Jonathan Verdun, is the sole decision-maker on these matters. This document provides analysis to inform that decision, not to make it.

---

**Disclaimer:** This document is for informational purposes only and does not constitute legal advice. Consult qualified legal professionals before making decisions about intellectual property, licensing, or regulatory compliance.

---

**Version:** 1.0 · **Updated:** 2025-11-27 · **Author:** Technical analysis for Jonathan Verdun
