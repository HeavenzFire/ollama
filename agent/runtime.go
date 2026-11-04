package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

const (
	// MaxActionsPerCycle bounds the amount of work a single model plan can request.
	MaxActionsPerCycle = 16
)

// Observation is a normalized sensory event received from an adapter.
// Payload is source data; adapters should minimize secrets and personal data.
type Observation struct {
	ID         string          `json:"id"`
	Source     string          `json:"source"`
	Kind       string          `json:"kind"`
	Payload    json.RawMessage `json:"payload"`
	ObservedAt time.Time       `json:"observed_at"`
}

// ProposedAction is a typed request produced by cognition.
// Arguments must be JSON and are interpreted only by a registered actuator.
type ProposedAction struct {
	ID         string          `json:"id"`
	Tool       string          `json:"tool"`
	Capability string          `json:"capability"`
	Arguments  json.RawMessage `json:"arguments"`
}

// Plan is the bounded output of one cognition cycle.
type Plan struct {
	Summary string           `json:"summary"`
	Actions []ProposedAction `json:"actions,omitempty"`
}

// MemoryRecord is an append-only event in a body's memory stream.
type MemoryRecord struct {
	ID        string          `json:"id"`
	Kind      string          `json:"kind"`
	Data      json.RawMessage `json:"data"`
	CreatedAt time.Time       `json:"created_at"`
}

// ActionResult records the disposition of a proposed action.
type ActionDisposition string

const (
	ActionExecuted         ActionDisposition = "executed"
	ActionDenied           ActionDisposition = "denied"
	ActionApprovalRequired ActionDisposition = "approval_required"
)

type ActionOutcome struct {
	ActionID    string            `json:"action_id"`
	Tool        string            `json:"tool"`
	Disposition ActionDisposition `json:"disposition"`
	Output      json.RawMessage   `json:"output,omitempty"`
	Reason      string            `json:"reason,omitempty"`
}

type CycleResult struct {
	BodyID   string          `json:"body_id"`
	Summary  string          `json:"summary"`
	Outcomes []ActionOutcome `json:"outcomes,omitempty"`
}

// BodyStore persists body identity and lifecycle state.
type BodyStore interface {
	Get(ctx context.Context, id string) (Body, error)
	Put(ctx context.Context, body Body) error
}

// MemoryStore loads and appends a body's durable memory records.
type MemoryStore interface {
	Load(ctx context.Context, bodyID string) ([]MemoryRecord, error)
	Append(ctx context.Context, bodyID string, record MemoryRecord) error
}

// Cognition converts observations and prior memory into a bounded plan.
// Implementations may call Ollama's local inference API or another model provider.
type Cognition interface {
	Think(ctx context.Context, body Body, observation Observation, memory []MemoryRecord) (Plan, error)
}

// Authorization is evaluated for every proposed action.
type Authorization string

const (
	AuthorizationAllow            Authorization = "allow"
	AuthorizationDeny             Authorization = "deny"
	AuthorizationApprovalRequired Authorization = "approval_required"
)

type Policy interface {
	Authorize(ctx context.Context, body Body, action ProposedAction) (Authorization, string, error)
}

// Actuator executes one already-authorized action. Implementations should use
// action.ID as an idempotency key to prevent duplicate effects on retries.
type Actuator interface {
	Execute(ctx context.Context, body Body, action ProposedAction) (json.RawMessage, error)
}

// Runtime coordinates one bounded perception-to-action cycle.
// It deliberately does not create goroutines or execute unbounded agent loops.
type Runtime struct {
	Bodies   BodyStore
	Memory   MemoryStore
	Cognition Cognition
	Policy   Policy
	Actuator Actuator
	Now      func() time.Time
}

func (r *Runtime) now() time.Time {
	if r.Now != nil {
		return r.Now().UTC()
	}
	return time.Now().UTC()
}

// RunCycle processes exactly one observation and at most one proposed plan.
// Actions requiring approval are recorded but never executed by this method.
func (r *Runtime) RunCycle(ctx context.Context, bodyID string, observation Observation) (CycleResult, error) {
	if r == nil || r.Bodies == nil || r.Memory == nil || r.Cognition == nil || r.Policy == nil || r.Actuator == nil {
		return CycleResult{}, fmt.Errorf("agent runtime dependencies are incomplete")
	}
	if bodyID == "" {
		return CycleResult{}, fmt.Errorf("body id is required")
	}
	if observation.ID == "" || observation.Source == "" || observation.Kind == "" {
		return CycleResult{}, fmt.Errorf("observation id, source, and kind are required")
	}
	if len(observation.Payload) == 0 || !json.Valid(observation.Payload) {
		return CycleResult{}, fmt.Errorf("observation payload must be valid JSON")
	}
	if observation.ObservedAt.IsZero() {
		observation.ObservedAt = r.now()
	} else {
		observation.ObservedAt = observation.ObservedAt.UTC()
	}

	body, err := r.Bodies.Get(ctx, bodyID)
	if err != nil {
		return CycleResult{}, fmt.Errorf("load body: %w", err)
	}
	if body.State != StateActive {
		return CycleResult{}, fmt.Errorf("body %q is not active (state %q)", body.ID, body.State)
	}

	memory, err := r.Memory.Load(ctx, body.ID)
	if err != nil {
		return CycleResult{}, fmt.Errorf("load memory: %w", err)
	}
	if err := r.append(ctx, body.ID, "observation", observation.ID, observation); err != nil {
		return CycleResult{}, fmt.Errorf("persist observation before cognition: %w", err)
	}

	plan, err := r.Cognition.Think(ctx, body, observation, memory)
	if err != nil {
		return CycleResult{}, fmt.Errorf("cognition: %w", err)
	}
	if err := r.append(ctx, body.ID, "plan", observation.ID+"/plan", plan); err != nil {
		return CycleResult{}, fmt.Errorf("persist plan before action: %w", err)
	}
	if len(plan.Actions) > MaxActionsPerCycle {
		return CycleResult{BodyID: body.ID, Summary: plan.Summary}, fmt.Errorf("plan has %d actions; maximum per cycle is %d", len(plan.Actions), MaxActionsPerCycle)
	}
	seenActionIDs := make(map[string]struct{}, len(plan.Actions))
	for _, action := range plan.Actions {
		if action.ID == "" {
			continue
		}
		if _, exists := seenActionIDs[action.ID]; exists {
			return CycleResult{BodyID: body.ID, Summary: plan.Summary}, fmt.Errorf("plan contains duplicate action id %q", action.ID)
		}
		seenActionIDs[action.ID] = struct{}{}
	}

	result := CycleResult{BodyID: body.ID, Summary: plan.Summary}
	for _, action := range plan.Actions {
		outcome := ActionOutcome{ActionID: action.ID, Tool: action.Tool}
		if action.ID == "" || action.Tool == "" || action.Capability == "" || len(action.Arguments) == 0 || !json.Valid(action.Arguments) {
			outcome.Disposition = ActionDenied
			outcome.Reason = "action is missing required fields or has invalid JSON arguments"
		} else if !body.HasCapability(action.Capability) {
			outcome.Disposition = ActionDenied
			outcome.Reason = "body does not declare the required capability"
		} else {
			decision, reason, policyErr := r.Policy.Authorize(ctx, body, action)
			if policyErr != nil {
				return result, fmt.Errorf("authorize action %q: %w", action.ID, policyErr)
			}
			switch decision {
			case AuthorizationAllow:
				output, executeErr := r.Actuator.Execute(ctx, body, action)
				if executeErr != nil {
					return result, fmt.Errorf("execute action %q: %w", action.ID, executeErr)
				}
				if len(output) > 0 && !json.Valid(output) {
					return result, fmt.Errorf("actuator returned invalid JSON for action %q", action.ID)
				}
				outcome.Disposition = ActionExecuted
				outcome.Output = output
			case AuthorizationApprovalRequired:
				outcome.Disposition = ActionApprovalRequired
				outcome.Reason = reason
			default:
				outcome.Disposition = ActionDenied
				outcome.Reason = reason
			}
		}

		result.Outcomes = append(result.Outcomes, outcome)
		if err := r.append(ctx, body.ID, "action_outcome", action.ID, outcome); err != nil {
			return result, fmt.Errorf("persist outcome for action %q: %w", action.ID, err)
		}
	}
	return result, nil
}

func (r *Runtime) append(ctx context.Context, bodyID, kind, id string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return r.Memory.Append(ctx, bodyID, MemoryRecord{
		ID:        id + "/" + kind,
		Kind:      kind,
		Data:      data,
		CreatedAt: r.now(),
	})
}
