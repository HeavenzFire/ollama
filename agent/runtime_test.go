package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

type testBodyStore struct{ body Body }
func (s testBodyStore) Get(context.Context, string) (Body, error) { return s.body, nil }
func (s testBodyStore) Put(context.Context, Body) error { return nil }

type testMemory struct{ records []MemoryRecord }
func (m *testMemory) Load(context.Context, string) ([]MemoryRecord, error) {
	return append([]MemoryRecord(nil), m.records...), nil
}
func (m *testMemory) Append(_ context.Context, _ string, record MemoryRecord) error {
	m.records = append(m.records, record)
	return nil
}

type testCognition struct{ plan Plan }
func (c testCognition) Think(context.Context, Body, Observation, []MemoryRecord) (Plan, error) {
	return c.plan, nil
}

type testPolicy struct{ decision Authorization }
func (p testPolicy) Authorize(context.Context, Body, ProposedAction) (Authorization, string, error) {
	return p.decision, "approval gate", nil
}

type testActuator struct{ calls int }
func (a *testActuator) Execute(context.Context, Body, ProposedAction) (json.RawMessage, error) {
	a.calls++
	return json.RawMessage(`{"ok":true}`), nil
}

func activeTestBody(t *testing.T) Body {
	t.Helper()
	body, err := NewBody("body-1", "Builder", "qwen2.5", []string{"files.write"}, time.Now())
	if err != nil { t.Fatal(err) }
	if err := body.Transition(StateReady, time.Now()); err != nil { t.Fatal(err) }
	if err := body.Transition(StateActive, time.Now()); err != nil { t.Fatal(err) }
	return body
}

func TestRunCycleRequiresApprovalBeforeExecution(t *testing.T) {
	body := activeTestBody(t)
	memory := &testMemory{}
	actuator := &testActuator{}
	runtime := Runtime{
		Bodies: testBodyStore{body: body},
		Memory: memory,
		Cognition: testCognition{plan: Plan{Summary: "write requested file", Actions: []ProposedAction{{
			ID: "action-1", Tool: "file.write", Capability: "files.write",
			Arguments: json.RawMessage(`{"path":"notes.txt","content":"hello"}`),
		}}}},
		Policy: testPolicy{decision: AuthorizationApprovalRequired},
		Actuator: actuator,
	}
	result, err := runtime.RunCycle(context.Background(), body.ID, Observation{
		ID: "obs-1", Source: "user", Kind: "request", Payload: json.RawMessage(`{"intent":"write a note"}`),
	})
	if err != nil { t.Fatal(err) }
	if len(result.Outcomes) != 1 || result.Outcomes[0].Disposition != ActionApprovalRequired {
		t.Fatalf("expected approval-required outcome, got %+v", result.Outcomes)
	}
	if actuator.calls != 0 {
		t.Fatalf("actuator ran before approval: %d calls", actuator.calls)
	}
	if len(memory.records) != 3 {
		t.Fatalf("expected observation, plan, and action outcome records; got %d", len(memory.records))
	}
}

func TestRunCycleDeniesUndeclaredCapability(t *testing.T) {
	body := activeTestBody(t)
	actuator := &testActuator{}
	runtime := Runtime{
		Bodies: testBodyStore{body: body},
		Memory: &testMemory{},
		Cognition: testCognition{plan: Plan{Actions: []ProposedAction{{
			ID: "action-2", Tool: "shell.exec", Capability: "system.shell",
			Arguments: json.RawMessage(`{"command":"whoami"}`),
		}}}},
		Policy: testPolicy{decision: AuthorizationAllow},
		Actuator: actuator,
	}
	result, err := runtime.RunCycle(context.Background(), body.ID, Observation{
		ID: "obs-2", Source: "user", Kind: "request", Payload: json.RawMessage(`{"intent":"inspect"}`),
	})
	if err != nil { t.Fatal(err) }
	if result.Outcomes[0].Disposition != ActionDenied || actuator.calls != 0 {
		t.Fatalf("undeclared capability was not denied: result=%+v calls=%d", result, actuator.calls)
	}
}

func TestRunCycleRejectsInactiveBody(t *testing.T) {
	body, err := NewBody("body-1", "Builder", "qwen2.5", nil, time.Now())
	if err != nil { t.Fatal(err) }
	actuator := &testActuator{}
	runtime := Runtime{
		Bodies: testBodyStore{body: body}, Memory: &testMemory{},
		Cognition: testCognition{}, Policy: testPolicy{decision: AuthorizationAllow},
		Actuator: actuator,
	}
	_, err = runtime.RunCycle(context.Background(), body.ID, Observation{
		ID: "obs", Source: "user", Kind: "request", Payload: json.RawMessage(`{}`),
	})
	if err == nil || errors.Is(err, context.Canceled) {
		t.Fatal("inactive body must be rejected")
	}
	if actuator.calls != 0 {
		t.Fatal("inactive body must not execute actions")
	}
}

func TestRunCycleRejectsOversizedPlan(t *testing.T) {
	body := activeTestBody(t)
	actions := make([]ProposedAction, DefaultMaxActionsPerCycle+1)
	for i := range actions {
		actions[i] = ProposedAction{
			ID: "action-" + time.Duration(i+1).String(),
			Tool: "file.write",
			Capability: "files.write",
			Arguments: json.RawMessage(`{"path":"x"}`),
		}
	}
	actuator := &testActuator{}
	runtime := Runtime{
		Bodies: testBodyStore{body: body},
		Memory: &testMemory{},
		Cognition: testCognition{plan: Plan{Actions: actions}},
		Policy: testPolicy{decision: AuthorizationAllow},
		Actuator: actuator,
	}
	_, err := runtime.RunCycle(context.Background(), body.ID, Observation{
		ID: "obs-limit", Source: "user", Kind: "request", Payload: json.RawMessage(`{}`),
	})
	if err == nil {
		t.Fatal("oversized plan must be rejected")
	}
	if actuator.calls != 0 {
		t.Fatalf("oversized plan executed %d actions", actuator.calls)
	}
}

func TestRunCycleRejectsDuplicateActionIDs(t *testing.T) {
	body := activeTestBody(t)
	action := ProposedAction{
		ID: "duplicate", Tool: "file.write", Capability: "files.write",
		Arguments: json.RawMessage(`{"path":"x"}`),
	}
	actuator := &testActuator{}
	runtime := Runtime{
		Bodies: testBodyStore{body: body},
		Memory: &testMemory{},
		Cognition: testCognition{plan: Plan{Actions: []ProposedAction{action, action}}},
		Policy: testPolicy{decision: AuthorizationAllow},
		Actuator: actuator,
	}
	_, err := runtime.RunCycle(context.Background(), body.ID, Observation{
		ID: "obs-duplicate", Source: "user", Kind: "request", Payload: json.RawMessage(`{}`),
	})
	if err == nil {
		t.Fatal("duplicate action IDs must be rejected")
	}
	if actuator.calls != 0 {
		t.Fatalf("duplicate action plan executed %d actions", actuator.calls)
	}
}
