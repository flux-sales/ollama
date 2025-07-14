package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// Dummy client for testing purposes
type Message struct {
	Content string `json:"content"`
}
type ChatResponse struct {
	Message    Message `json:"message"`
	Done       bool    `json:"done,omitempty"`
	DoneReason string  `json:"done_reason,omitempty"`
}

// Simulates errors
type testError struct {
	message    string
	statusCode int
}

func (e testError) Error() string {
	return e.message
}

func TestClientFromEnvironment(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "1.2.3.4:1234")
	client, err := ClientFromEnvironment()
	if err != nil {
		t.Fatal("unexpected error:", err)
	}
	want := "http://1.2.3.4:1234"
	if client.base.String() != want {
		t.Errorf("expected %q, got %q", want, client.base.String())
	}
}

func TestClientDo(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "abc",
			"success": true,
		})
	}))
	defer ts.Close()

	client := NewClient(&url.URL{Scheme: "http", Host: ts.Listener.Addr().String()}, http.DefaultClient)

	var resp map[string]any
	err := client.do(context.Background(), http.MethodPost, "/test", nil, &resp)
	if err != nil {
		t.Fatal("unexpected error:", err)
	}

	if resp["id"] != "abc" || !resp["success"].(bool) {
		t.Error("unexpected response:", resp)
	}
}

func TestClientStreamSuccess(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, _ := w.(http.Flusher)
		json.NewEncoder(w).Encode(ChatResponse{Message: Message{Content: "chunk 1"}})
		flusher.Flush()
		json.NewEncoder(w).Encode(ChatResponse{Message: Message{Content: "chunk 2"}, Done: true})
		flusher.Flush()
	}))
	defer ts.Close()

	client := NewClient(&url.URL{Scheme: "http", Host: ts.Listener.Addr().String()}, http.DefaultClient)

	var received []string
	err := client.stream(context.Background(), http.MethodPost, "/stream", nil, func(chunk []byte) error {
		var cr ChatResponse
		if err := json.Unmarshal(chunk, &cr); err != nil {
			return err
		}
		received = append(received, cr.Message.Content)
		return nil
	})

	if err != nil {
		t.Fatal("unexpected error:", err)
	}
	if len(received) != 2 || received[0] != "chunk 1" || received[1] != "chunk 2" {
		t.Error("unexpected streamed data:", received)
	}
}

func TestClientStreamError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "stream failed"})
	}))
	defer ts.Close()

	client := NewClient(&url.URL{Scheme: "http", Host: ts.Listener.Addr().String()}, http.DefaultClient)

	err := client.stream(context.Background(), http.MethodPost, "/stream", nil, func(chunk []byte) error {
		return nil
	})

	if err == nil || !strings.Contains(err.Error(), "stream failed") {
		t.Errorf("expected stream error, got: %v", err)
	}
}
