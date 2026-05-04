package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	noop "go.opentelemetry.io/otel/trace/noop"
)

// ─── Helpers ──────────────────────────────────────────────────────────────

func TestMain(m *testing.M) {
	otelTracer = noop.NewTracerProvider().Tracer("test")
	os.Exit(m.Run())
}

// sseBuffer accumulates raw bytes and can split into individual SSE events
type sseBuffer struct {
	raw bytes.Buffer
}

func (b *sseBuffer) Write(p []byte) (int, error) {
	return b.raw.Write(p)
}

func (b *sseBuffer) events() []map[string]interface{} {
	text := b.raw.String()
	var out []map[string]interface{}
	for _, line := range strings.Split(text, "\n\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimSpace(line[6:])
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(payload), &m); err == nil {
			out = append(out, m)
		}
	}
	return out
}

func getEventByType(events []map[string]interface{}, tp string) map[string]interface{} {
	for _, e := range events {
		if t, ok := e["type"].(string); ok && t == tp {
			return e
		}
	}
	return nil
}

func getEventWithTypeAndIndex(events []map[string]interface{}, tp string, idx int) map[string]interface{} {
	for _, e := range events {
		if t, ok := e["type"].(string); ok && t == tp {
			if i, ok := e["index"].(float64); ok && int(i) == idx {
				return e
			}
		}
	}
	return nil
}

// numericVal extracts a numeric value from interface{} (handles int, float64, etc.)
func numericVal(v interface{}) float64 {
	switch n := v.(type) {
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case float64:
		return n
	case float32:
		return float64(n)
	default:
		return 0
	}
}

// ─── extractText ──────────────────────────────────────────────────────────

func TestExtractText(t *testing.T) {
	tests := []struct {
		name string
		in   interface{}
		want string
	}{
		{"string", "hello", "hello"},
		{"nil", nil, ""},
		{"single text block", map[string]interface{}{"type": "text", "text": "hi"}, "hi"},
		{"two text blocks",
			[]interface{}{
				map[string]interface{}{"type": "text", "text": "A"},
				map[string]interface{}{"type": "text", "text": "B"},
			},
			"A\nB",
		},
		{"tool_result string",
			[]interface{}{
				map[string]interface{}{"type": "tool_result", "tool_use_id": "tu_1", "content": "42"},
			},
			"42",
		},
		{"tool_result blocks",
			[]interface{}{
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "tu_1",
					"content": []interface{}{
						map[string]interface{}{"type": "text", "text": "ok"},
					},
				},
			},
			"ok",
		},
		{"mixed text + tool_result",
			[]interface{}{
				map[string]interface{}{"type": "text", "text": "res:"},
				map[string]interface{}{"type": "tool_result", "tool_use_id": "tu_1", "content": "x"},
			},
			"res:\nx",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := extractText(tt.in); got != tt.want {
				t.Errorf("extractText() = %q, want %q", got, tt.want)
			}
		})
	}
}

// ─── extractToolUses ──────────────────────────────────────────────────────

func TestExtractToolUses(t *testing.T) {
	if out := extractToolUses("just a string"); out != nil {
		t.Errorf("expected nil for non-block input, got %v", out)
	}
	if out := extractToolUses(nil); out != nil {
		t.Errorf("expected nil for nil input, got %v", out)
	}

	blocks := []interface{}{
		map[string]interface{}{"type": "text", "text": "thinking"},
		map[string]interface{}{"type": "tool_use", "id": "tu_1", "name": "search",
			"input": map[string]interface{}{"query": "hello"}},
		map[string]interface{}{"type": "tool_use", "id": "tu_2", "name": "calc",
			"input": map[string]interface{}{"expr": "2+2"}},
	}

	out := extractToolUses(blocks)
	if len(out) != 2 {
		t.Fatalf("expected 2 tool uses, got %d", len(out))
	}
	if out[0].ID != "tu_1" || out[0].Name != "search" {
		t.Errorf("first tool_use = %+v", out[0])
	}
	if out[1].ID != "tu_2" || out[1].Name != "calc" {
		t.Errorf("second tool_use = %+v", out[1])
	}
}

// ─── extractToolResults ───────────────────────────────────────────────────

func TestExtractToolResults(t *testing.T) {
	blocks := []interface{}{
		map[string]interface{}{
			"type":        "tool_result",
			"tool_use_id": "tu_1",
			"content":     "data",
		},
		map[string]interface{}{"type": "text", "text": "ignored"},
		map[string]interface{}{
			"type":        "tool_result",
			"tool_use_id": "tu_2",
			"content": []interface{}{
				map[string]interface{}{"type": "text", "text": "nested"},
			},
		},
	}

	out := extractToolResults(blocks)
	if len(out) != 2 {
		t.Fatalf("expected 2 results, got %d", len(out))
	}
	if out[0].ToolUseID != "tu_1" || out[0].Content != "data" {
		t.Errorf("first = %+v", out[0])
	}
	if out[1].ToolUseID != "tu_2" || out[1].Content != "nested" {
		t.Errorf("second = %+v", out[1])
	}
}

// ─── getSystemText ────────────────────────────────────────────────────────

func TestGetSystemText(t *testing.T) {
	tests := []struct {
		name string
		in   interface{}
		want string
	}{
		{"string", "be helpful", "be helpful"},
		{"nil", nil, ""},
		{"blocks",
			[]interface{}{
				map[string]interface{}{"type": "text", "text": "A"},
				map[string]interface{}{"type": "text", "text": "B"},
				map[string]interface{}{"type": "image", "source": "x"},
			},
			"AB",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getSystemText(tt.in); got != tt.want {
				t.Errorf("getSystemText() = %q, want %q", got, tt.want)
			}
		})
	}
}

// ─── convertTools ─────────────────────────────────────────────────────────

func TestConvertToolsAnthropic(t *testing.T) {
	raw := []json.RawMessage{
		json.RawMessage(`{"name":"search","description":"Search","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}`),
	}
	out := convertTools(raw)
	if len(out) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(out))
	}
	if out[0].Function.Name != "search" {
		t.Errorf("name = %q", out[0].Function.Name)
	}
	if out[0].Function.Description != "Search" {
		t.Errorf("desc = %q", out[0].Function.Description)
	}
	if out[0].Type != "function" {
		t.Errorf("type = %q, want function", out[0].Type)
	}
}

func TestConvertToolsOpenAI(t *testing.T) {
	raw := []json.RawMessage{
		json.RawMessage(`{"type":"function","function":{"name":"calc","description":"Calculate","parameters":{"type":"object"}}}`),
	}
	out := convertTools(raw)
	if len(out) != 1 || out[0].Function.Name != "calc" {
		t.Errorf("unexpected result: %+v", out)
	}
}

func TestConvertToolsMixed(t *testing.T) {
	raw := []json.RawMessage{
		json.RawMessage(`{"name":"a","description":"A","input_schema":{"type":"object"}}`),
		json.RawMessage(`{"type":"function","function":{"name":"b","description":"B","parameters":{"type":"object"}}}`),
	}
	out := convertTools(raw)
	if len(out) != 2 {
		t.Fatalf("expected 2, got %d", len(out))
	}
	if out[0].Function.Name != "a" || out[1].Function.Name != "b" {
		t.Errorf("names = %s, %s", out[0].Function.Name, out[1].Function.Name)
	}
}

func TestConvertToolsInvalid(t *testing.T) {
	raw := []json.RawMessage{
		json.RawMessage(`{"garbage": true}`),
		json.RawMessage(`{}`),
	}
	out := convertTools(raw)
	if len(out) != 0 {
		t.Errorf("expected 0 tools, got %d", len(out))
	}
}

// ─── anthropicToOpenAI ────────────────────────────────────────────────────

func TestAnthropicToOpenAIBasic(t *testing.T) {
	req := &AnthropicRequest{
		Model:    "claude-sonnet-4-20250514",
		Messages: []AnthropicMessage{{Role: "user", Content: "Hello"}},
	}
	openai := anthropicToOpenAI(context.Background(), req)

	if openai.Model != vllmModel {
		t.Errorf("model = %q, want %q", openai.Model, vllmModel)
	}
	if !openai.Stream {
		t.Error("expected stream=true")
	}
	if openai.StreamOptions == nil || !openai.StreamOptions.IncludeUsage {
		t.Error("expected stream_options.include_usage=true")
	}
	if len(openai.Messages) != 1 || openai.Messages[0].Content != "Hello" {
		t.Errorf("messages = %+v", openai.Messages)
	}
}

func TestAnthropicToOpenAIWithSystem(t *testing.T) {
	tests := []struct {
		name  string
		sys   interface{}
		want  string
		count int // total messages (system + user)
	}{
		{"string", "Be nice", "Be nice", 2},
		{"blocks",
			[]interface{}{
				map[string]interface{}{"type": "text", "text": "A"},
				map[string]interface{}{"type": "text", "text": "B"},
			},
			"AB", 2},
		{"empty", "", "", 1},
		{"whitespace", "   ", "", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &AnthropicRequest{
				Model:    "claude-sonnet-4-20250514",
				System:   tt.sys,
				Messages: []AnthropicMessage{{Role: "user", Content: "hi"}},
			}
			openai := anthropicToOpenAI(context.Background(), req)
			if len(openai.Messages) != tt.count {
				t.Fatalf("expected %d messages, got %d", tt.count, len(openai.Messages))
			}
			if tt.count == 2 {
				if openai.Messages[0].Role != "system" {
					t.Errorf("first msg role = %q, want system", openai.Messages[0].Role)
				}
				if openai.Messages[0].Content != tt.want {
					t.Errorf("system content = %q, want %q", openai.Messages[0].Content, tt.want)
				}
			}
		})
	}
}

func TestAnthropicToOpenAIWithToolUses(t *testing.T) {
	req := &AnthropicRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []AnthropicMessage{
			{Role: "user", Content: "Hi"},
			{Role: "assistant", Content: []interface{}{
				map[string]interface{}{"type": "tool_use", "id": "tu_1", "name": "search",
					"input": map[string]interface{}{"q": "test"}},
			}},
		},
	}
	openai := anthropicToOpenAI(context.Background(), req)
	if len(openai.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(openai.Messages))
	}
	tc := openai.Messages[1].ToolCalls
	if len(tc) != 1 || tc[0].Function.Name != "search" {
		t.Errorf("tool call = %+v", tc)
	}
	if openai.Messages[1].Content != "" {
		t.Errorf("content should be empty for tool-use message, got %q", openai.Messages[1].Content)
	}
}

func TestAnthropicToOpenAIWithToolResults(t *testing.T) {
	req := &AnthropicRequest{
		Model: "claude-sonnet-4-20250514",
		Messages: []AnthropicMessage{
			{Role: "user", Content: "Hi"},
			{Role: "assistant", Content: []interface{}{
				map[string]interface{}{"type": "tool_use", "id": "tu_1", "name": "s", "input": map[string]interface{}{}},
			}},
			{Role: "user", Content: []interface{}{
				map[string]interface{}{"type": "tool_result", "tool_use_id": "tu_1", "content": "result"},
			}},
		},
	}
	openai := anthropicToOpenAI(context.Background(), req)
	if len(openai.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(openai.Messages))
	}
	last := openai.Messages[2]
	if last.Role != "tool" || last.ToolCallID != "tu_1" || last.Content != "result" {
		t.Errorf("last msg = %+v", last)
	}
}

func TestAnthropicToOpenAIPassesParams(t *testing.T) {
	temp := 0.7
	topP := 0.9
	topK := 5.0
	req := &AnthropicRequest{
		Model:         "claude-sonnet-4-20250514",
		Messages:      []AnthropicMessage{{Role: "user", Content: "hi"}},
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		MaxTokens:     2048,
		StopSequences: []string{"END"},
	}
	openai := anthropicToOpenAI(context.Background(), req)
	if openai.Temperature == nil || *openai.Temperature != 0.7 {
		t.Errorf("temperature = %v", openai.Temperature)
	}
	if openai.TopP == nil || *openai.TopP != 0.9 {
		t.Errorf("topP = %v", openai.TopP)
	}
	if openai.TopK == nil || *openai.TopK != 5.0 {
		t.Errorf("topK = %v", openai.TopK)
	}
	if openai.MaxTokens != 2048 {
		t.Errorf("maxTokens = %d", openai.MaxTokens)
	}
	if len(openai.Stop) != 1 || openai.Stop[0] != "END" {
		t.Errorf("stop = %v", openai.Stop)
	}
}

// ─── openaiToAnthropicResp ────────────────────────────────────────────────

func TestOpenAIToAnthropicText(t *testing.T) {
	resp := &OpenAIResponse{
		ID:    "c1",
		Model: "m",
		Choices: []OpenAIChoice{{
			Index:   0,
			Message: OpenAIMessage{Role: "assistant", Content: "hi"},
		}},
		Usage: &OpenAIUsage{PromptTokens: 10, CompletionTokens: 5},
	}
	out := openaiToAnthropicResp(resp)
	if out["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v", out["stop_reason"])
	}
	usage := out["usage"].(map[string]interface{})
	if numericVal(usage["input_tokens"]) != 10 || numericVal(usage["output_tokens"]) != 5 {
		t.Errorf("usage = %v", usage)
	}
}

func TestOpenAIToAnthropicToolUse(t *testing.T) {
	resp := &OpenAIResponse{
		ID:    "c1",
		Model: "m",
		Choices: []OpenAIChoice{{
			Index: 0,
			Message: OpenAIMessage{
				Role:    "assistant",
				Content: "",
				ToolCalls: []OpenAIToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: OpenAIFuncCall{
						Name:      "s",
						Arguments: `{"q":"t"}`,
					},
				}},
			},
		}},
		Usage: &OpenAIUsage{PromptTokens: 20, CompletionTokens: 15},
	}
	out := openaiToAnthropicResp(resp)
	if out["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", out["stop_reason"])
	}
	content := out["content"].([]map[string]interface{})
	if len(content) != 1 || content[0]["type"] != "tool_use" || content[0]["id"] != "call_1" {
		t.Errorf("content = %+v", content)
	}
}

func TestOpenAIToAnthropicBothTextAndTool(t *testing.T) {
	resp := &OpenAIResponse{
		ID:    "c1",
		Model: "m",
		Choices: []OpenAIChoice{{
			Index: 0,
			Message: OpenAIMessage{
				Role:    "assistant",
				Content: "searching",
				ToolCalls: []OpenAIToolCall{{
					ID:   "c1",
					Type: "function",
					Function: OpenAIFuncCall{
						Name:      "s",
						Arguments: `{}`,
					},
				}},
			},
		}},
		Usage: &OpenAIUsage{PromptTokens: 100, CompletionTokens: 50},
	}
	out := openaiToAnthropicResp(resp)
	content := out["content"].([]map[string]interface{})
	if len(content) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(content))
	}
	if content[0]["type"] != "text" || content[1]["type"] != "tool_use" {
		t.Errorf("block types = %v, %v", content[0]["type"], content[1]["type"])
	}
}

func TestOpenAIToAnthropicNoUsage(t *testing.T) {
	resp := &OpenAIResponse{
		ID:      "c1",
		Model:   "m",
		Choices: []OpenAIChoice{{
			Index:   0,
			Message: OpenAIMessage{Role: "assistant", Content: "hi"},
		}},
	}
	out := openaiToAnthropicResp(resp)
	usage := out["usage"].(map[string]interface{})
	if numericVal(usage["input_tokens"]) != 0 || numericVal(usage["output_tokens"]) != 0 {
		t.Errorf("expected zero usage, got %v", usage)
	}
}

// ─── newStreamState & findOrCreateTC ──────────────────────────────────────

func TestNewStreamState(t *testing.T) {
	st := time.Now()
	s := newStreamState("msg_1", 42, st)
	if s.msgID != "msg_1" || s.inputTokens != 42 || s.startTime != st {
		t.Errorf("state = %+v", s)
	}
	if s.thinkingIdx != 0 || s.textIdx != 1 || s.tcNextIdx != 2 {
		t.Errorf("indices wrong: think=%d text=%d tc=%d", s.thinkingIdx, s.textIdx, s.tcNextIdx)
	}
}

func TestFindOrCreateTC(t *testing.T) {
	s := newStreamState("m", 0, time.Now())

	if pos := s.findOrCreateTC(0, "s", "c1"); pos != 0 {
		t.Errorf("expected 0, got %d", pos)
	}

	if pos := s.findOrCreateTC(0, "s", "c1"); pos != 0 {
		t.Errorf("expected 0, got %d", pos)
	}
	if len(s.tcs) != 1 {
		t.Errorf("expected 1 tc, got %d", len(s.tcs))
	}

	if pos := s.findOrCreateTC(1, "x", "c2"); pos != 1 {
		t.Errorf("expected 1, got %d", pos)
	}
	if s.tcNextIdx != 4 {
		t.Errorf("tcNextIdx = %d, want 4 (started at 2, created 2 new: 2→3→4)", s.tcNextIdx)
	}
}

// ─── processVLLMLine ──────────────────────────────────────────────────────

func TestProcessVLLMContent(t *testing.T) {
	st := time.Now().Add(-2 * time.Second)
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	tokens, _, done := processVLLMLine("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}", s, &buf)
	if done || tokens != 1 || !s.text || !s.gotFirstToken {
		t.Errorf("done=%v tokens=%d text=%v first=%v", done, tokens, s.text, s.gotFirstToken)
	}

	evts := buf.events()
	if len(evts) < 2 {
		t.Fatalf("expected >=2 events, got %d", len(evts))
	}
	if evts[0]["type"] != "content_block_start" {
		t.Errorf("first event = %v", evts[0]["type"])
	}
}

func TestProcessVLLMReasoningContent(t *testing.T) {
	st := time.Now().Add(-1 * time.Second)
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	tokens, _, done := processVLLMLine("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}", s, &buf)
	if done || tokens != 1 || !s.thinking || !s.gotFirstToken {
		t.Errorf("done=%v tokens=%d think=%v first=%v", done, tokens, s.thinking, s.gotFirstToken)
	}
}

func TestProcessVLLMReasoningField(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	tokens, _, done := processVLLMLine("data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\"}}]}", s, &buf)
	if tokens != 1 || done || !s.thinking {
		t.Errorf("tokens=%d done=%v thinking=%v", tokens, done, s.thinking)
	}
}

func TestProcessVLLMToolCall(t *testing.T) {
	st := time.Now().Add(-1 * time.Second)
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	processVLLMLine("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"search\"}}]}}]}", s, &buf)
	tokens, _, done := processVLLMLine("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\":\\\"\"}}]}}]}", s, &buf)
	if done || tokens != 1 || !s.gotFirstToken {
		t.Errorf("done=%v tokens=%d first=%v", done, tokens, s.gotFirstToken)
	}
	if len(s.tcs) != 1 || s.tcs[0].name != "search" {
		t.Errorf("tc state = %+v", s.tcs)
	}
}

func TestProcessVLLMMultipleToolCalls(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	tokens, _, done := processVLLMLine("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"a\",\"arguments\":\"{}\"}},{\"index\":1,\"id\":\"c2\",\"function\":{\"name\":\"b\",\"arguments\":\"{}\"}}]}}]}", s, &buf)
	if done || tokens != 2 {
		t.Errorf("done=%v tokens=%d", done, tokens)
	}
	if len(s.tcs) != 2 {
		t.Errorf("tcs len = %d, want 2", len(s.tcs))
	}
}

func TestProcessVLLMFinishReason(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	_, stop, done := processVLLMLine("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}", s, &buf)
	if !done || stop != "end_turn" {
		t.Errorf("done=%v stop=%q", done, stop)
	}
}

func TestProcessVLLMFinishReasonWithTools(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	s.tcs = append(s.tcs, tcState{vllmIdx: 0, name: "s", id: "c1", idx: 2})
	var buf sseBuffer

	_, stop, done := processVLLMLine("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}", s, &buf)
	if !done || stop != "tool_use" {
		t.Errorf("done=%v stop=%q, want true, tool_use", done, stop)
	}
}

func TestProcessVLLMUsageParsing(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 0, st)
	var buf sseBuffer

	processVLLMLine("data: {\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":10,\"total_tokens\":60},\"choices\":[{\"delta\":{}}]}", s, &buf)
	if s.inputTokens != 50 {
		t.Errorf("inputTokens = %d, want 50", s.inputTokens)
	}
}

func TestProcessVLLMDone(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	_, stop, done := processVLLMLine("data: [DONE]", s, &buf)
	if !done || stop != "end_turn" {
		t.Errorf("done=%v stop=%q", done, stop)
	}
}

func TestProcessVLLMEmpty(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	_, _, done := processVLLMLine("data: ", s, &buf)
	if done {
		t.Error("expected not done for empty data")
	}
}

func TestProcessVLLMRoleOnly(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	tokens, _, done := processVLLMLine("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}", s, &buf)
	if done || tokens != 0 || s.gotFirstToken {
		t.Errorf("done=%v tokens=%d first=%v", done, tokens, s.gotFirstToken)
	}
}

func TestProcessVLLMReasoningAndContent(t *testing.T) {
	st := time.Now().Add(-2 * time.Second)
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	tokens, _, done := processVLLMLine("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\",\"content\":\"hello\"}}]}", s, &buf)
	if done || tokens != 2 || !s.thinking || !s.text || !s.gotFirstToken {
		t.Errorf("done=%v tokens=%v think=%v text=%v first=%v", done, tokens, s.thinking, s.text, s.gotFirstToken)
	}
}

func TestProcessVLLMMalformed(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	_, _, done := processVLLMLine("data: {invalid}", s, &buf)
	if done {
		t.Error("expected not done for malformed JSON")
	}
}

func TestProcessVLLMNoChoices(t *testing.T) {
	st := time.Now()
	s := newStreamState("m", 100, st)
	var buf sseBuffer

	_, _, done := processVLLMLine("data: {\"id\":\"chatcmpl-1\"}", s, &buf)
	if done {
		t.Error("expected not done for no-choices payload")
	}
}

// ─── finishStream ─────────────────────────────────────────────────────────

func TestFinishStreamTextWithTPS(t *testing.T) {
	startTime := time.Now().Add(-3 * time.Second)
	firstToken := startTime.Add(1 * time.Second) // 1s after start = 2s ago

	s := newStreamState("m", 100, startTime)
	s.text = true
	s.gotFirstToken = true
	s.firstTokenAt = firstToken

	var buf sseBuffer
	finishStream(&buf, s, 50, "end_turn")

	evts := buf.events()
	delta := getEventByType(evts, "message_delta")
	if delta == nil {
		t.Fatal("expected message_delta event")
	}
	stop := getEventByType(evts, "message_stop")
	if stop == nil {
		t.Fatal("expected message_stop event")
	}

	usage := delta["usage"].(map[string]interface{})
	if numericVal(usage["input_tokens"]) != 100 || numericVal(usage["output_tokens"]) != 50 {
		t.Errorf("tokens: input=%v output=%v", usage["input_tokens"], usage["output_tokens"])
	}
	if prefillTPS := numericVal(usage["prefill_tokens_per_second"]); prefillTPS < 50 || prefillTPS > 200 {
		t.Errorf("prefill TPS = %v, expected ~100", prefillTPS)
	}
	if decodeTPS := numericVal(usage["decode_tokens_per_second"]); decodeTPS < 5 || decodeTPS > 100 {
		t.Errorf("decode TPS = %v, expected ~25", decodeTPS)
	}
	if numericVal(usage["prefill_duration_seconds"]) <= 0 {
		t.Error("expected positive prefill_duration_seconds")
	}
	if numericVal(usage["decode_duration_seconds"]) <= 0 {
		t.Error("expected positive decode_duration_seconds")
	}
}

func TestFinishStreamNoTiming(t *testing.T) {
	s := newStreamState("m", 50, time.Now())
	s.text = true

	var buf sseBuffer
	finishStream(&buf, s, 10, "end_turn")

	delta := getEventByType(buf.events(), "message_delta")
	usage := delta["usage"].(map[string]interface{})
	if numericVal(usage["input_tokens"]) != 50 || numericVal(usage["output_tokens"]) != 10 {
		t.Errorf("tokens: input=%v output=%v", usage["input_tokens"], usage["output_tokens"])
	}
	if _, ok := usage["prefill_tokens_per_second"]; ok {
		t.Error("should NOT have TPS when no timing data")
	}
}

func TestFinishStreamThinking(t *testing.T) {
	s := newStreamState("m", 0, time.Now())
	s.thinking = true
	s.text = true

	var buf sseBuffer
	finishStream(&buf, s, 5, "end_turn")

	evts := buf.events()
	thinkStop := getEventWithTypeAndIndex(evts, "content_block_stop", 0)
	textStop := getEventWithTypeAndIndex(evts, "content_block_stop", 1)
	if thinkStop == nil || textStop == nil {
		t.Errorf("missing content_block_stop events: think=%v text=%v", thinkStop, textStop)
	}
}

func TestFinishStreamToolUse(t *testing.T) {
	s := newStreamState("m", 0, time.Now())
	s.tcs = append(s.tcs, tcState{vllmIdx: 0, name: "s", id: "c1", idx: 2, opened: true})

	var buf sseBuffer
	finishStream(&buf, s, 10, "tool_use")

	tcStop := getEventWithTypeAndIndex(buf.events(), "content_block_stop", 2)
	if tcStop == nil {
		t.Error("missing content_block_stop for tool call")
	}
}

func TestFinishStreamEmpty(t *testing.T) {
	s := newStreamState("m", 0, time.Now())

	var buf sseBuffer
	finishStream(&buf, s, 0, "end_turn")

	evts := buf.events()
	if getEventByType(evts, "content_block_start") == nil {
		t.Error("expected placeholder content_block_start")
	}
	if getEventByType(evts, "content_block_delta") == nil {
		t.Error("expected placeholder content_block_delta")
	}
}

func TestFinishStreamTPSCalc(t *testing.T) {
	startTime := time.Now().Add(-3 * time.Second)
	firstToken := startTime.Add(1 * time.Second) // 1s after start = 2s ago

	s := newStreamState("m", 100, startTime)
	s.text = true
	s.gotFirstToken = true
	s.firstTokenAt = firstToken

	var buf sseBuffer
	finishStream(&buf, s, 50, "end_turn")

	usage := getEventByType(buf.events(), "message_delta")["usage"].(map[string]interface{})
	prefillTPS := numericVal(usage["prefill_tokens_per_second"])
	decodeTPS := numericVal(usage["decode_tokens_per_second"])

	if prefillTPS < 50 || prefillTPS > 200 {
		t.Errorf("prefill TPS = %v, expected ~100", prefillTPS)
	}
	if decodeTPS < 5 || decodeTPS > 100 {
		t.Errorf("decode TPS = %v, expected ~25", decodeTPS)
	}
}

// ─── sendEvent ────────────────────────────────────────────────────────────

func TestSendEvent(t *testing.T) {
	var buf sseBuffer
	sendEvent(&buf, map[string]interface{}{"type": "test", "value": 42})
	raw := buf.raw.String()
	if !strings.Contains(raw, "data: {\"type\":\"test\",\"value\":42}\n\n") {
		t.Errorf("unexpected output: %q", raw)
	}
}

// ─── handleMessages HTTP handler ───────────────────────────────────────────

func TestHandleMessagesMethods(t *testing.T) {
	tests := []struct {
		method string
		body   string
		code   int
	}{
		{"GET", "", http.StatusMethodNotAllowed},
		{"OPTIONS", "", http.StatusOK},
		{"POST", "not json", http.StatusBadRequest},
		{"PUT", "", http.StatusMethodNotAllowed},
	}
	for _, tt := range tests {
		t.Run(tt.method, func(t *testing.T) {
			var body io.Reader
			if tt.body != "" {
				body = strings.NewReader(tt.body)
			}
			req := httptest.NewRequest(tt.method, "/v1/messages", body)
			w := httptest.NewRecorder()
			handleMessages(w, req)
			if w.Code != tt.code {
				t.Errorf("status = %d, want %d", w.Code, tt.code)
			}
		})
	}
}

func TestHandleMessagesClaudeModelRejected(t *testing.T) {
	tests := []string{"claude-3-haiku-20240307", "claude-code-internal", "claude-opus-4"}
	for _, model := range tests {
		t.Run(model, func(t *testing.T) {
			body := fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"hi"}]}`, model)
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
			w := httptest.NewRecorder()
			handleMessages(w, req)
			if w.Code != http.StatusNotFound {
				t.Errorf("status = %d, want 404", w.Code)
			}
		})
	}
}

func TestHandleMessagesNonClaude(t *testing.T) {
	body := `{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	handleMessages(w, req)
	// No real vLLM, so expect error
	if w.Code != http.StatusBadGateway && w.Code != http.StatusInternalServerError {
		t.Logf("status = %d (expected 502 or 500 since no vLLM running)", w.Code)
	}
}

func TestHandleModels(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	handleModels(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	data, ok := resp["data"].([]interface{})
	if !ok || len(data) != 1 {
		t.Errorf("expected 1 model in data array, got %+v", resp)
	}
}

// ─── flushingRW ───────────────────────────────────────────────────────────

func TestFlushingRW(t *testing.T) {
	rec := httptest.NewRecorder()
	fw := &flushingRW{w: rec, f: rec}
	n, err := fw.Write([]byte("hello"))
	if err != nil || n != 5 {
		t.Errorf("Write returned n=%d err=%v", n, err)
	}
	if rec.Body.String() != "hello" {
		t.Errorf("body = %q, want hello", rec.Body.String())
	}
}

// ─── collectVLLM with mock server ─────────────────────────────────────────

func TestCollectVLLM(t *testing.T) {
	orig := vllmURL
	defer func() { vllmURL = orig }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"test-m\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"He\"}}]}\n\n"))
		w.Write([]byte("data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":3,\"total_tokens\":18}}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	vllmURL = server.URL
	body := `{"model":"t","messages":[{"role":"user","content":"hi"}]}`
	resp, err := collectVLLM(context.Background(), []byte(body))
	if err != nil {
		t.Fatalf("collectVLLM error: %v", err)
	}
	if resp.Choices[0].Message.Content != "Hello" {
		t.Errorf("content = %q, want Hello", resp.Choices[0].Message.Content)
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 15 {
		t.Errorf("prompt_tokens = %v, want 15", resp.Usage)
	}
}

func TestCollectVLLMWithReasoning(t *testing.T) {
	orig := vllmURL
	defer func() { vllmURL = orig }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	vllmURL = server.URL
	body := `{"model":"t","messages":[{"role":"user","content":"hi"}]}`
	resp, err := collectVLLM(context.Background(), []byte(body))
	if err != nil {
		t.Fatalf("collectVLLM error: %v", err)
	}
	if resp.Choices[0].Message.Content != "answer" {
		t.Errorf("content = %q, want answer", resp.Choices[0].Message.Content)
	}
}

func TestCollectVLLMReasoningOnly(t *testing.T) {
	orig := vllmURL
	defer func() { vllmURL = orig }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"full response\"}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"total_tokens\":13}}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	vllmURL = server.URL
	body := `{"model":"t","messages":[{"role":"user","content":"hi"}]}`
	resp, err := collectVLLM(context.Background(), []byte(body))
	if err != nil {
		t.Fatalf("collectVLLM error: %v", err)
	}
	if resp.Choices[0].Message.Content != "full response" {
		t.Errorf("content = %q, want 'full response'", resp.Choices[0].Message.Content)
	}
}

func TestCollectVLLMWithToolCalls(t *testing.T) {
	orig := vllmURL
	defer func() { vllmURL = orig }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"s\"}}]}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\":\\\"t\\\"}\"}}]}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":2,\"total_tokens\":22}}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	vllmURL = server.URL
	body := `{"model":"t","messages":[{"role":"user","content":"hi"}]}`
	resp, err := collectVLLM(context.Background(), []byte(body))
	if err != nil {
		t.Fatalf("collectVLLM error: %v", err)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Choices[0].Message.ToolCalls))
	}
	tc := resp.Choices[0].Message.ToolCalls[0]
	if tc.ID != "c1" || tc.Function.Name != "s" {
		t.Errorf("tool call = %+v", tc)
	}
}

func TestCollectVLLMNoUsageChunk(t *testing.T) {
	orig := vllmURL
	defer func() { vllmURL = orig }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	vllmURL = server.URL
	body := `{"model":"t","messages":[{"role":"user","content":"hi"}]}`
	resp, err := collectVLLM(context.Background(), []byte(body))
	if err != nil {
		t.Fatalf("collectVLLM error: %v", err)
	}
	if resp.Choices[0].Message.Content != "hi" {
		t.Errorf("content = %q, want hi", resp.Choices[0].Message.Content)
	}
	if resp.Usage.PromptTokens != 0 {
		t.Errorf("prompt_tokens = %d, want 0", resp.Usage.PromptTokens)
	}
}

func TestCollectVLLMConnectionError(t *testing.T) {
	orig := vllmURL
	defer func() { vllmURL = orig }()

	vllmURL = "http://localhost:59999"
	_, err := collectVLLM(context.Background(), []byte(`{}`))
	if err == nil {
		t.Error("expected error for unreachable vLLM")
	}
}

// ─── HTTP handler with mock vLLM ──────────────────────────────────────────

func TestHandleMessagesNonStreamingWithMock(t *testing.T) {
	orig := vllmURL
	defer func() { vllmURL = orig }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"test-m\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	vllmURL = server.URL

	body := `{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	handleMessages(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v\nbody: %s", err, w.Body.String())
	}
	if resp["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v", resp["stop_reason"])
	}
	usage := resp["usage"].(map[string]interface{})
	if numericVal(usage["input_tokens"]) != 10 {
		t.Errorf("input_tokens = %v, want 10", usage["input_tokens"])
	}
}
