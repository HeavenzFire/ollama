package agent

import (
	"testing"
	"time"
)

func TestNewBodyNormalizesCapabilities(t *testing.T) {
	now := time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC)
	body, err := NewBody("body-1", "Builder", "qwen2.5", []string{"files.read", "files.read", "tasks.execute"}, now)
	if err != nil {
		t.Fatal(err)
	}
	if body.State != StateCreated || body.Revision != 1 {
		t.Fatalf("unexpected initial state: %+v", body)
	}
	if len(body.Capabilities) != 2 || body.Capabilities[0] != "files.read" || body.Capabilities[1] != "tasks.execute" {
		t.Fatalf("capabilities were not normalized: %#v", body.Capabilities)
	}
	if !body.HasCapability("files.read") || body.HasCapability("system.root") {
		t.Fatal("capability membership is incorrect")
	}
}

func TestBodyLifecycleRejectsInvalidTransitions(t *testing.T) {
	now := time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC)
	body, err := NewBody("body-1", "Builder", "qwen2.5", nil, now)
	if err != nil {
		t.Fatal(err)
	}
	if err := body.Transition(StateActive, now.Add(time.Second)); err == nil {
		t.Fatal("created body must not transition directly to active")
	}
	for _, state := range []LifecycleState{StateReady, StateActive, StatePaused, StateReady, StateStopped} {
		if err := body.Transition(state, now.Add(time.Duration(body.Revision+1)*time.Second)); err != nil {
			t.Fatalf("transition to %q failed: %v", state, err)
		}
	}
	if err := body.Transition(StateReady, now.Add(time.Hour)); err == nil {
		t.Fatal("stopped body must be terminal")
	}
}
