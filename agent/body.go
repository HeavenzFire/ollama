package agent

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// LifecycleState describes the operational state of a digital body.
// A body is a persistent agent identity; it is not equivalent to a model process.
type LifecycleState string

const (
	StateCreated LifecycleState = "created"
	StateReady   LifecycleState = "ready"
	StateActive  LifecycleState = "active"
	StatePaused  LifecycleState = "paused"
	StateFaulted LifecycleState = "faulted"
	StateStopped LifecycleState = "stopped"
)

// Body is the versionable identity and lifecycle record for an agent.
type Body struct {
	ID           string         `json:"id"`
	Name         string         `json:"name"`
	Model        string         `json:"model"`
	State        LifecycleState `json:"state"`
	Capabilities []string       `json:"capabilities,omitempty"`
	CreatedAt    time.Time      `json:"created_at"`
	UpdatedAt    time.Time      `json:"updated_at"`
	Revision     uint64         `json:"revision"`
}

// NewBody creates a body in the created state with normalized capabilities.
func NewBody(id, name, model string, capabilities []string, now time.Time) (Body, error) {
	id, name, model = strings.TrimSpace(id), strings.TrimSpace(name), strings.TrimSpace(model)
	if id == "" {
		return Body{}, fmt.Errorf("body id is required")
	}
	if name == "" {
		return Body{}, fmt.Errorf("body name is required")
	}
	if model == "" {
		return Body{}, fmt.Errorf("body model is required")
	}
	if now.IsZero() {
		now = time.Now().UTC()
	} else {
		now = now.UTC()
	}

	caps := make([]string, 0, len(capabilities))
	seen := make(map[string]struct{}, len(capabilities))
	for _, capability := range capabilities {
		capability = strings.TrimSpace(capability)
		if capability == "" {
			return Body{}, fmt.Errorf("capability cannot be empty")
		}
		if _, ok := seen[capability]; ok {
			continue
		}
		seen[capability] = struct{}{}
		caps = append(caps, capability)
	}
	sort.Strings(caps)

	return Body{
		ID:           id,
		Name:         name,
		Model:        model,
		State:        StateCreated,
		Capabilities: caps,
		CreatedAt:    now,
		UpdatedAt:    now,
		Revision:     1,
	}, nil
}

// HasCapability reports whether this body has a declared capability.
// A declaration is not authorization: Policy must still approve each action.
func (b Body) HasCapability(capability string) bool {
	capability = strings.TrimSpace(capability)
	if capability == "" {
		return false
	}
	for _, granted := range b.Capabilities {
		if granted == capability {
			return true
		}
	}
	return false
}

// Transition applies a validated lifecycle transition.
func (b *Body) Transition(next LifecycleState, now time.Time) error {
	if b == nil {
		return fmt.Errorf("body is nil")
	}
	if now.IsZero() {
		now = time.Now().UTC()
	} else {
		now = now.UTC()
	}

	allowed := false
	switch b.State {
	case StateCreated:
		allowed = next == StateReady || next == StateStopped
	case StateReady:
		allowed = next == StateActive || next == StatePaused || next == StateStopped
	case StateActive:
		allowed = next == StateReady || next == StatePaused || next == StateFaulted || next == StateStopped
	case StatePaused:
		allowed = next == StateReady || next == StateFaulted || next == StateStopped
	case StateFaulted:
		allowed = next == StateReady || next == StateStopped
	case StateStopped:
		allowed = false
	}
	if !allowed {
		return fmt.Errorf("invalid body lifecycle transition: %q -> %q", b.State, next)
	}

	b.State = next
	b.UpdatedAt = now
	b.Revision++
	return nil
}
