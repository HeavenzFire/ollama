# HeavenzFire Agentic Bodies — Architecture v0.1

## Purpose

Evolve this Ollama fork from a model inference service into an agent execution substrate for the HeavenzFire Unified Field. Preserve Ollama's model loading, scheduling, inference, and existing API compatibility. Add agency as a distinct layer rather than treating a model prompt as an agent.

A **digital body** is a persistent, addressable software entity with identity, lifecycle, model assignment, memory, perception adapters, capability-bounded actuators, health state, and an auditable event history. This is an engineering definition; it does not imply biological life, consciousness, or sentience.

## Layer model

1. **Inference substrate** — existing Ollama model discovery, loading, scheduling, chat, embeddings, and multimodal inference.
2. **Body identity and lifecycle** — stable body IDs, versioned manifests, lifecycle transitions, health and resource budgets.
3. **Perception fabric** — normalized observations from user input, files, local system telemetry, network peers, images, audio, and approved connectors.
4. **Cognition** — model/provider adapters that convert observations and memory into bounded plans. Record provider, model, prompt/schema version, and evidence references.
5. **Memory** — append-only episodic records, searchable semantic memory, working context, checkpoints, retention and export policies.
6. **Action fabric** — typed tool adapters. Every action requires a declared body capability and a policy decision; high-impact actions require explicit human approval.
7. **Federation** — authenticated body-to-body messages and task handoffs through the Unified Field control plane.
8. **Embodiment interfaces** — optional voice, avatar, spatial display, robotics, and device adapters. A visual avatar is a representation, not the body runtime itself.

## First implementation slice

The `agent` Go package introduces:
- a body identity record and explicit lifecycle state machine;
- normalized capability declarations;
- observation, plan, memory, and action outcome contracts;
- one bounded perception-to-action cycle;
- a policy gate between model-proposed actions and actuators;
- a hard rule that approval-required actions are recorded but not executed.

The runtime is dependency-injected. This keeps storage, model providers, policy, and actuators replaceable and testable. The default action budget is 16 per cycle and can be raised explicitly by the runtime configuration. It does not yet expose HTTP routes, ship a durable production store, run background loops, or connect to physical devices.

## Security invariants

- Model output is untrusted input, never authority.
- Capability declarations are necessary but not sufficient: policy authorizes each action.
- Unknown actions, undeclared capabilities, inactive bodies, and malformed payloads fail closed.
- Approval-required actions never reach an actuator until a separate approval workflow authorizes them.
- Actuators should use action IDs as idempotency keys and return machine-readable receipts.
- Credentials are references to a local secret manager, never embedded in body manifests, memory records, or prompts.
- Observations and tool output are untrusted data and cannot modify policy or grant capabilities.
- Resource limits, cancellation, timeouts, and audit retention are mandatory before production deployment.

## Integration sequence

### Phase 1 — Body core
Implement and test identity, lifecycle, normalized observations, memory interfaces, bounded cognition cycles, capability checks, and action policy gates.

### Phase 2 — Durable continuity
Add an atomic local body registry, append-only event journal, checkpoint recovery, retention controls, export, and deletion receipts. Reuse or explicitly bridge the UnifiedCodebase task ledger rather than creating a competing task system.

### Phase 3 — Ollama inference adapter and API
Implement cognition against the existing `api.Client.Chat` interface, structured plan parsing, and versioned `/api/agents` endpoints. Preserve existing `/api/chat` and `/api/generate` behavior.

### Phase 4 — Perception and actuator adapters
Add user messages, local files, health/resource telemetry, and authenticated peer events as observation sources. Add narrowly scoped tools with policy-enforced permissions and human approval for privileged operations.

### Phase 5 — Federation and embodiment
Connect bodies to the Unified Field task ledger and authenticated message fabric. Add voice and spatial/3D representations as optional adapters. Physical actuators require a separate safety case and hardware-specific interlocks.

## Verification and compliance

The body core is not production-complete until tests cover:
- every valid and invalid lifecycle transition;
- inactive-body rejection;
- malformed observation and action rejection;
- undeclared capability denial;
- policy denial and approval-required non-execution;
- cancellation, timeout, duplicate action IDs, and idempotent retries;
- memory write failures before and after external side effects;
- checkpoint recovery and export/deletion behavior;
- untrusted observation/tool output attempting to influence policy.

Current status: **Phase 1 source added; runtime integration, durable storage, HTTP API, federation, and device embodiment remain unimplemented.** Do not describe the system as autonomous or production-ready until those phases have passed their acceptance tests.
