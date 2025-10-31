# COELANOX Security Model

## API Separation Strategy

COELANOX uses a two-tier security model.

## Access Tiers

### Tier 1: Frontend API (Public)
**Purpose**: Format translation to Universal IR
**Access**: Open to all developers
**Risk Level**: Low
**Use Cases**: Framework creators building translators
**Documentation**: [Frontend API Reference](./docs/frontend-api.md)

### Tier 2: Backend API (Partner-Only)  
**Purpose**: Hardware-specific execution and runtime integration
**Access**: Verified hardware vendors only
**Risk Level**: High - critical security boundary
**Use Cases**: Hardware vendors building execution backends
**Documentation**: [Backend API Reference](./docs/backend-api.md)
**Onboarding**: [Partner Onboarding Guide](./docs/PARTNER_ONBOARDING.md)

## Security Rationale

Backend APIs are restricted because they represent the most critical security boundary:
- Direct interface between hardware and runtime
- Memory management across hardware boundaries
- Performance optimization access
- Execution control flow

## Partner Requirements

Hardware vendors must:
- Complete security assessment
- Sign partnership agreement
- Undergo technical review
- Maintain secure implementation
- Report vulnerabilities promptly

## Contact

For partnership inquiries: partners@coelanox.ai
