package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
)

var (
	vllmURL   = os.Getenv("VLLM_URL")
	vllmModel = os.Getenv("VLLM_MODEL")
	proxyUser = os.Getenv("PROXY_USER")
)

func init() {
	if vllmURL == "" {
		vllmURL = "http://localhost:6006"
	}
	if vllmModel == "" {
		vllmModel = "Lorbus/Qwen3.6-27B-int4-AutoRound"
	}
	if proxyUser == "" {
		proxyUser = "anonymous"
	}
}

// ─── Anthropic request types ───────────────────────────────────────────────

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type AnthropicRequest struct {
	Model         string            `json:"model"`
	Messages      []AnthropicMessage `json:"messages"`
	System        interface{}       `json:"system,omitempty"`
	MaxTokens     int               `json:"max_tokens,omitempty"`
	Temperature   *float64          `json:"temperature,omitempty"`
	TopP          *float64          `json:"top_p,omitempty"`
	TopK          *float64          `json:"top_k,omitempty"`
	Stream        bool              `json:"stream,omitempty"`
	StopSequences []string          `json:"stop_sequences,omitempty"`
	Tools         []json.RawMessage `json:"tools,omitempty"`
}

// ─── OpenAI types ─────────────────────────────────────────────────────────

type OpenAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type OpenAIRequest struct {
	Model         string             `json:"model"`
	Messages      []OpenAIMessage    `json:"messages"`
	MaxTokens     int                `json:"max_tokens,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *float64           `json:"top_k,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	StreamOptions *OpenAIStreamOptions `json:"stream_options,omitempty"`
	Stop          []string           `json:"stop,omitempty"`
	Tools         []OpenAITool       `json:"tools,omitempty"`
}

type OpenAIMessage struct {
	Role       string           `json:"role,omitempty"`
	Content    string           `json:"content,omitempty"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Name       string           `json:"name,omitempty"`
}

type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFuncSpec `json:"function"`
}

type OpenAIFuncSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type OpenAIToolCall struct {
	ID       string         `json:"id"`
	Type     string         `json:"type"`
	Function OpenAIFuncCall `json:"function"`
}

type OpenAIFuncCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIResponse struct {
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Choices []OpenAIChoice  `json:"choices"`
	Usage   *OpenAIUsage    `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int              `json:"index"`
	Message      OpenAIMessage    `json:"message,omitempty"`
	Delta        OpenAIMessageDelta `json:"delta,omitempty"`
	FinishReason string           `json:"finish_reason,omitempty"`
}

type OpenAIMessageDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	Reasoning string          `json:"reasoning,omitempty"`
	ToolCalls []ToolCallDelta `json:"tool_calls,omitempty"`
}

type ToolCallDelta struct {
	Index    int       `json:"index"`
	ID       string    `json:"id,omitempty"`
	Type     string    `json:"type,omitempty"`
	Function FuncDelta `json:"function,omitempty"`
}

type FuncDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// ─── Content extraction (Anthropic blocks → plain text) ───────────────────

func extractText(c interface{}) string {
	switch v := c.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, b := range v {
			if m, ok := b.(map[string]interface{}); ok {
				switch m["type"] {
				case "text":
					if t, ok := m["text"].(string); ok {
						parts = append(parts, t)
					}
				case "tool_result":
					if cc, ok := m["content"].(string); ok {
						parts = append(parts, cc)
					} else if ca, ok := m["content"].([]interface{}); ok {
						for _, cb := range ca {
							if cm, ok := cb.(map[string]interface{}); ok && cm["type"] == "text" {
								if t, ok := cm["text"].(string); ok {
									parts = append(parts, t)
								}
							}
						}
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	case map[string]interface{}:
		switch v["type"] {
		case "text":
			return v["text"].(string)
		case "tool_result":
			return extractText(v["content"])
		}
	}
	return ""
}

type toolUseInfo struct {
	ID    string
	Name  string
	Input map[string]interface{}
}

func extractToolUses(c interface{}) []toolUseInfo {
	blocks, ok := c.([]interface{})
	if !ok {
		return nil
	}
	var out []toolUseInfo
	for _, b := range blocks {
		m, ok := b.(map[string]interface{})
		if !ok || m["type"] != "tool_use" {
			continue
		}
		tu := toolUseInfo{
			ID:   fmt.Sprintf("%v", m["id"]),
			Name: fmt.Sprintf("%v", m["name"]),
		}
		if inp, ok := m["input"].(map[string]interface{}); ok {
			tu.Input = inp
		}
		out = append(out, tu)
	}
	return out
}

type toolResultInfo struct {
	ToolUseID string
	Content   string
}

func extractToolResults(c interface{}) []toolResultInfo {
	blocks, ok := c.([]interface{})
	if !ok {
		return nil
	}
	var out []toolResultInfo
	for _, b := range blocks {
		m, ok := b.(map[string]interface{})
		if !ok || m["type"] != "tool_result" {
			continue
		}
		tr := toolResultInfo{
			ToolUseID: fmt.Sprintf("%v", m["tool_use_id"]),
			Content:   extractText(m["content"]),
		}
		out = append(out, tr)
	}
	return out
}

func getSystemText(sys interface{}) string {
	switch s := sys.(type) {
	case string:
		return s
	case []interface{}:
		var p []string
		for _, b := range s {
			if m, ok := b.(map[string]interface{}); ok && m["type"] == "text" {
				if t, ok := m["text"].(string); ok {
					p = append(p, t)
				}
			}
		}
		return strings.Join(p, "")
	}
	return ""
}

// ─── Tool conversion (Anthropic → OpenAI) ─────────────────────────────────

func convertTools(raw []json.RawMessage) []OpenAITool {
	var tools []OpenAITool
	for _, r := range raw {
		var t map[string]interface{}
		if err := json.Unmarshal(r, &t); err != nil {
			continue
		}

		// Handle Anthropic format: {"name": "...", "description": "...", "input_schema": {...}}
		if name, ok := t["name"].(string); ok {
			desc, _ := t["description"].(string)
			params, _ := t["input_schema"].(map[string]interface{})
			paramsJSON, _ := json.Marshal(params)
			tools = append(tools, OpenAITool{
				Type: "function",
				Function: OpenAIFuncSpec{
					Name:        name,
					Description: desc,
					Parameters:  json.RawMessage(paramsJSON),
				},
			})
			continue
		}

		// Handle OpenAI format: {"type": "function", "function": {...}}
		if t["type"] != "function" {
			continue
		}
		fn, ok := t["function"].(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := fn["name"].(string)
		desc, _ := fn["description"].(string)
		params, _ := fn["parameters"].(map[string]interface{})
		paramsJSON, _ := json.Marshal(params)
		tools = append(tools, OpenAITool{
			Type: "function",
			Function: OpenAIFuncSpec{
				Name:        name,
				Description: desc,
				Parameters:  json.RawMessage(paramsJSON),
			},
		})
	}
	return tools
}

// ─── Translate Anthropic → OpenAI ─────────────────────────────────────────

func anthropicToOpenAI(ctx context.Context, req *AnthropicRequest) *OpenAIRequest {
	_, span := otelTracer.Start(ctx, "proxy.translate.request")
	defer span.End()

	openai := &OpenAIRequest{
		Model:       vllmModel,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		Stream:      true,
		Stop:        req.StopSequences,
	}

	if len(req.Tools) > 0 {
		openai.Tools = convertTools(req.Tools)
	}

	for _, m := range req.Messages {
		plain := extractText(m.Content)
		tu := extractToolUses(m.Content)
		tr := extractToolResults(m.Content)

		if len(tr) > 0 {
			for _, t := range tr {
				openai.Messages = append(openai.Messages, OpenAIMessage{
					Role:       "tool",
					ToolCallID: t.ToolUseID,
					Content:    t.Content,
				})
			}
			continue
		}

		om := OpenAIMessage{Role: m.Role, Content: plain}
		if len(tu) > 0 {
			var tcs []OpenAIToolCall
			for _, tu := range tu {
				args, _ := json.Marshal(tu.Input)
				tcs = append(tcs, OpenAIToolCall{
					ID:   tu.ID,
					Type: "function",
					Function: OpenAIFuncCall{
						Name:      tu.Name,
						Arguments: string(args),
					},
				})
			}
			om.ToolCalls = tcs
			om.Content = ""
		}
		openai.Messages = append(openai.Messages, om)
	}

	if req.System != nil {
		sysText := getSystemText(req.System)
		if strings.TrimSpace(sysText) != "" {
			openai.Messages = append([]OpenAIMessage{
				{Role: "system", Content: sysText},
			}, openai.Messages...)
		}
	}

	// Always request usage data in final SSE chunk for input token count & timing
	openai.StreamOptions = &OpenAIStreamOptions{IncludeUsage: true}

	return openai
}

// ─── OpenAI → Anthropic (non-streaming) ───────────────────────────────────

func openaiToAnthropicResp(resp *OpenAIResponse) map[string]interface{} {
	var blocks []map[string]interface{}
	msg := resp.Choices[0].Message
	if msg.Content != "" {
		blocks = append(blocks, map[string]interface{}{"type": "text", "text": msg.Content})
	}
	for _, tc := range msg.ToolCalls {
		var input map[string]interface{}
		json.Unmarshal([]byte(tc.Function.Arguments), &input)
		blocks = append(blocks, map[string]interface{}{
			"type": "tool_use", "id": tc.ID, "name": tc.Function.Name, "input": input,
		})
	}
	if blocks == nil {
		blocks = []map[string]interface{}{}
	}

	usage := map[string]interface{}{"input_tokens": 0, "output_tokens": 0}
	if resp.Usage != nil {
		usage["input_tokens"] = resp.Usage.PromptTokens
		usage["output_tokens"] = resp.Usage.CompletionTokens
	}

	stopReason := "end_turn"
	if len(msg.ToolCalls) > 0 {
		stopReason = "tool_use"
	}

	return map[string]interface{}{
		"id": resp.ID, "type": "message", "role": "assistant",
		"model": resp.Model, "content": blocks,
		"stop_reason": stopReason, "stop_sequence": nil, "usage": usage,
	}
}

// ─── vLLM via http.Client ─────────────────────────────────────────────────

var vllmClient = &http.Client{
	Timeout: 120 * time.Second,
	Transport: &http.Transport{
		DisableKeepAlives:  true,
		ForceAttemptHTTP2:  false,
		MaxIdleConns:       0,
		DisableCompression: true,
	},
}

func requestVLLM(ctx context.Context, body []byte) (*http.Response, error) {
	_, span := otelTracer.Start(ctx, "proxy.forward.vllm")
	defer span.End()

	req, err := http.NewRequest("POST", vllmURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		span.SetStatus(codes.Error, "create request failed")
		return nil, fmt.Errorf("create request: %w", err)
	}
	// Inject trace context into outbound request headers
	carrier := propagation.HeaderCarrier{}
	otel.GetTextMapPropagator().Inject(ctx, carrier)
	for k, vs := range carrier {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-local")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Connection", "close")

	resp, err := vllmClient.Do(req)
	if err != nil {
		span.SetStatus(codes.Error, "request failed")
		return nil, fmt.Errorf("request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		span.SetStatus(codes.Error, fmt.Sprintf("vLLM returned %d", resp.StatusCode))
		resp.Body.Close()
		return nil, fmt.Errorf("vLLM returned %d: %s", resp.StatusCode, resp.Status)
	}
	return resp, nil
}

// ─── Collect (non-streaming mode) ─────────────────────────────────────────

func collectVLLM(ctx context.Context, body []byte) (*OpenAIResponse, error) {
	resp, err := requestVLLM(ctx, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var text, reasoning, id, model = "", "", "", vllmModel
	var toolCalls []OpenAIToolCall
	tokens := 0
	inputTokens := 0

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		ds := strings.TrimSpace(line[6:])
		if ds == "[DONE]" || ds == "" {
			continue
		}
		var p map[string]interface{}
		if err := json.Unmarshal([]byte(ds), &p); err != nil {
			continue
		}
		if id == "" {
			if v, ok := p["id"].(string); ok {
				id = v
			}
		}
		// Parse usage from final chunk (when stream_options.include_usage is set)
		if usageRaw, ok := p["usage"].(map[string]interface{}); ok {
			if pt, ok := usageRaw["prompt_tokens"].(float64); ok {
				inputTokens = int(pt)
			}
			if ct, ok := usageRaw["completion_tokens"].(float64); ok {
				tokens = int(ct) // Use vLLM's authoritative count
			}
		}
		if model == vllmModel {
			if v, ok := p["model"].(string); ok {
				model = v
			}
		}
		choices, ok := p["choices"].([]interface{})
		if !ok || len(choices) == 0 {
			continue
		}
		c, ok := choices[0].(map[string]interface{})
		if !ok {
			continue
		}
		d, ok := c["delta"].(map[string]interface{})
		if !ok {
			continue
		}
		if t, ok := d["content"].(string); ok {
			text += t
			tokens++
		}
		// Collect reasoning - append to text since Anthropic doesn't have reasoning in non-streaming
		if r, ok := d["reasoning"].(string); ok {
			reasoning += r
			tokens++
		}
		if r, ok := d["reasoning_content"].(string); ok {
			reasoning += r
			tokens++
		}
		if tcArr, ok := d["tool_calls"].([]interface{}); ok && len(tcArr) > 0 {
			for _, tcRaw := range tcArr {
				tcd, ok := tcRaw.(map[string]interface{})
				if !ok {
					continue
				}
				idxF, _ := tcd["index"].(float64)
				idx := int(idxF)
				tcID, _ := tcd["id"].(string)
				funcDelta, ok := tcd["function"].(map[string]interface{})
				if !ok {
					continue
				}
				tcName, _ := funcDelta["name"].(string)
				tcArgs, _ := funcDelta["arguments"].(string)

				for len(toolCalls) <= idx {
					toolCalls = append(toolCalls, OpenAIToolCall{})
				}
				if tcID != "" {
					toolCalls[idx].ID = tcID
				}
				if tcName != "" {
					toolCalls[idx].Type = "function"
					toolCalls[idx].Function.Name = tcName
				}
				if tcArgs != "" {
					toolCalls[idx].Function.Arguments += tcArgs
					tokens++
				}
			}
		}
	}

	// If model only produced reasoning, use that as the text content
	if text == "" && reasoning != "" {
		text = reasoning
	}

	return &OpenAIResponse{
		ID: id, Model: model,
		Choices: []OpenAIChoice{{Index: 0, Message: OpenAIMessage{Role: "assistant", Content: text, ToolCalls: toolCalls}}},
		Usage: &OpenAIUsage{PromptTokens: inputTokens, CompletionTokens: tokens},
	}, nil
}

func handleCollected(ctx context.Context, w http.ResponseWriter, body []byte) (int, int) {
	_, span := otelTracer.Start(ctx, "proxy.collect")
	defer span.End()

	log.Println("[COLLECT] Starting")
	resp, err := collectVLLM(ctx, body)
	if err != nil {
		log.Printf("[COLLECT] Error: %v", err)
		span.SetStatus(codes.Error, "vLLM error")
		http.Error(w, fmt.Sprintf("vLLM: %v", err), http.StatusBadGateway)
		return 0, 0
	}
	log.Println("[COLLECT] Done")
	ar := openaiToAnthropicResp(resp)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ar)

	outputTokens := 0
	inputTokens := 0
	if resp.Usage != nil {
		outputTokens = resp.Usage.CompletionTokens
		inputTokens = resp.Usage.PromptTokens
	}
	span.SetAttributes(
		attribute.Int("llm.output.tokens", outputTokens),
		attribute.Int("llm.input.tokens", inputTokens),
		attribute.String("llm.response.model", resp.Model),
	)
	return outputTokens, inputTokens
}

// ─── Streaming state ──────────────────────────────────────────────────────

type tcState struct {
	vllmIdx int
	opened  bool
	name    string
	id      string
	args    string
	idx     int
}

type streamState struct {
	thinkingIdx    int
	thinking       bool
	thinkingTokens int
	textIdx        int
	text           bool
	textTokens     int
	tcNextIdx      int
	tcs            []tcState
	msgID          string
	inputTokens    int
	// Timing for TPS (tokens per second) estimation
	startTime     time.Time
	firstTokenAt  time.Time
	gotFirstToken bool
}

func newStreamState(msgID string, inputTokens int, startTime time.Time) *streamState {
	return &streamState{
		thinkingIdx: 0,
		textIdx:     1,
		tcNextIdx:   2,
		tcs:         make([]tcState, 0),
		msgID:       msgID,
		inputTokens: inputTokens,
		startTime:   startTime,
	}
}

func (s *streamState) findOrCreateTC(vllmIdx int, tcName, tcID string) int {
	for i, ts := range s.tcs {
		if ts.vllmIdx == vllmIdx {
			return i
		}
	}
	idx := s.tcNextIdx
	s.tcNextIdx++
	s.tcs = append(s.tcs, tcState{vllmIdx: vllmIdx, name: tcName, id: tcID, idx: idx})
	return len(s.tcs) - 1
}

// ─── Write to output ──────────────────────────────────────────────────────

type sseWriter interface {
	Write(p []byte) (int, error)
}

func sendEvent(w sseWriter, obj interface{}) {
	data, err := json.Marshal(obj)
	if err != nil {
		log.Printf("[STREAM] marshal error: %v", err)
		return
	}
	w.Write([]byte("data: " + string(data) + "\n\n"))
}

func finishStream(w sseWriter, s *streamState, totalTokens int, stopReason string) {
	if s.thinking {
		sendEvent(w, map[string]interface{}{"type": "content_block_stop", "index": s.thinkingIdx})
	}
	if s.text {
		sendEvent(w, map[string]interface{}{"type": "content_block_stop", "index": s.textIdx})
	}
	for _, tc := range s.tcs {
		sendEvent(w, map[string]interface{}{"type": "content_block_stop", "index": tc.idx})
	}
	if !s.thinking && !s.text && len(s.tcs) == 0 {
		sendEvent(w, map[string]interface{}{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]interface{}{"type": "text", "text": ""},
		})
		sendEvent(w, map[string]interface{}{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]interface{}{"type": "text_delta", "text": ""},
		})
		sendEvent(w, map[string]interface{}{"type": "content_block_stop", "index": 0})
	}

	// Build usage with input + output tokens
	usage := map[string]interface{}{
		"input_tokens":  s.inputTokens,
		"output_tokens": totalTokens,
	}

	// Calculate TPS for prefill (input) and decode (output)
	if s.gotFirstToken && !s.firstTokenAt.IsZero() && !s.startTime.IsZero() {
		prefillDuration := s.firstTokenAt.Sub(s.startTime).Seconds()
		decodeDuration := time.Since(s.firstTokenAt).Seconds()
		totalDuration := time.Since(s.startTime).Seconds()

		if prefillDuration > 0 {
			usage["prefill_tokens_per_second"] = float64(s.inputTokens) / prefillDuration
		}
		if decodeDuration > 0 {
			usage["decode_tokens_per_second"] = float64(totalTokens) / decodeDuration
		}
		if totalDuration > 0 {
			usage["total_tokens_per_second"] = float64(s.inputTokens+totalTokens) / totalDuration
		}
		usage["prefill_duration_seconds"] = prefillDuration
		usage["decode_duration_seconds"] = decodeDuration
		usage["total_duration_seconds"] = totalDuration
	}

	sendEvent(w, map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": usage,
	})
	sendEvent(w, map[string]interface{}{"type": "message_stop"})
}

// ─── Process one SSE line from vLLM ────────────────────────────────────────

func processVLLMLine(line string, s *streamState, w sseWriter) (int, string, bool) {
	// Returns: (tokensAdded, stopReason, streamDone)
	dataStr := strings.TrimSpace(line[6:])
	if dataStr == "[DONE]" {
		return 0, "end_turn", true
	}
	if dataStr == "" {
		return 0, "", false
	}

	var payload map[string]interface{}
	if err := json.Unmarshal([]byte(dataStr), &payload); err != nil {
		return 0, "", false
	}

	// Parse usage from final chunk (when stream_options.include_usage is set)
	if usageRaw, ok := payload["usage"].(map[string]interface{}); ok {
		if pt, ok := usageRaw["prompt_tokens"].(float64); ok {
			s.inputTokens = int(pt)
		}
		// total_tokens and completion_tokens also available if needed
	}

	choices, ok := payload["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return 0, "", false
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return 0, "", false
	}

	if finishReason, ok := choice["finish_reason"]; ok && finishReason != nil {
		stopReason := "end_turn"
		if len(s.tcs) > 0 {
			stopReason = "tool_use"
		}
		return 0, stopReason, true
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return 0, "", false
	}

	tokensAdded := 0

	// Handle tool_calls
	if tcArr, ok := delta["tool_calls"].([]interface{}); ok && len(tcArr) > 0 {
		for _, tcRaw := range tcArr {
			tcd, ok := tcRaw.(map[string]interface{})
			if !ok {
				continue
			}
			idxFloat, _ := tcd["index"].(float64)
			vllmIdx := int(idxFloat)
			tcID, _ := tcd["id"].(string)
			funcDelta, ok := tcd["function"].(map[string]interface{})
			if !ok {
				continue
			}
			tcName, _ := funcDelta["name"].(string)
			tcArgs, _ := funcDelta["arguments"].(string)

			pos := s.findOrCreateTC(vllmIdx, tcName, tcID)
			ts := &s.tcs[pos]
			if tcName != "" {
				ts.name = tcName
			}
			if tcID != "" {
				ts.id = tcID
			}
			if tcArgs != "" {
				ts.args += tcArgs
			}

			if !ts.opened {
				ts.opened = true
				sendEvent(w, map[string]interface{}{
					"type":  "content_block_start",
					"index": ts.idx,
					"content_block": map[string]interface{}{
						"type": "tool_use", "id": ts.id, "name": ts.name,
						"input": map[string]interface{}{},
					},
				})
			}
			if tcArgs != "" {
				sendEvent(w, map[string]interface{}{
					"type":  "content_block_delta",
					"index": ts.idx,
					"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": tcArgs},
				})
				tokensAdded++
				if !s.gotFirstToken {
					s.gotFirstToken = true
					s.firstTokenAt = time.Now()
				}
			}
		}
		return tokensAdded, "", false
	}

	// Handle reasoning
	reasoning := ""
	if rc, ok := delta["reasoning_content"]; ok {
		if s, ok := rc.(string); ok {
			reasoning = s
		}
	}
	if reasoning == "" {
		if r, ok := delta["reasoning"]; ok {
			if s, ok := r.(string); ok {
				reasoning = s
			}
		}
	}

	// Handle content
	content := ""
	if c, ok := delta["content"]; ok {
		if s, ok := c.(string); ok {
			content = s
		}
	}

	if reasoning != "" {
		if !s.thinking {
			sendEvent(w, map[string]interface{}{
				"type": "content_block_start",
				"index": s.thinkingIdx,
				"content_block": map[string]interface{}{"type": "thinking"},
			})
			s.thinking = true
		}
		sendEvent(w, map[string]interface{}{
			"type":  "content_block_delta",
			"index": s.thinkingIdx,
			"delta": map[string]interface{}{"type": "thinking_delta", "thinking": reasoning},
		})
		s.thinkingTokens++
		tokensAdded++
		if !s.gotFirstToken {
			s.gotFirstToken = true
			s.firstTokenAt = time.Now()
		}
	}

	if content != "" {
		if !s.text {
			sendEvent(w, map[string]interface{}{
				"type": "content_block_start",
				"index": s.textIdx,
				"content_block": map[string]interface{}{"type": "text", "text": ""},
			})
			s.text = true
		}
		sendEvent(w, map[string]interface{}{
			"type":  "content_block_delta",
			"index": s.textIdx,
			"delta": map[string]interface{}{"type": "text_delta", "text": content},
		})
		s.textTokens++
		tokensAdded++
		if !s.gotFirstToken {
			s.gotFirstToken = true
			s.firstTokenAt = time.Now()
		}
	}

	return tokensAdded, "", false
}

// ─── Handle streaming ─────────────────────────────────────────────────────

func handleStreaming(w http.ResponseWriter, ctx context.Context, openaiJSON []byte) (int, int) {
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		log.Println("[PROXY] hijacker not available, using fallback")
		return handleStreamingFallback(w, ctx, openaiJSON)
	}

	conn, bufrw, err := hijacker.Hijack()
	if err != nil {
		log.Printf("[PROXY] hijack error: %v, using fallback", err)
		return handleStreamingFallback(w, ctx, openaiJSON)
	}
	defer conn.Close()

	// Disable Nagle's algorithm for immediate writes
	if tcp, ok := conn.(*net.TCPConn); ok {
		tcp.SetNoDelay(true)
	}

	// Write HTTP response headers directly to raw socket
	conn.Write([]byte(
		"HTTP/1.1 200 OK\r\n" +
			"Content-Type: text/event-stream\r\n" +
			"Cache-Control: no-cache\r\n" +
			"Connection: close\r\n" +
			"\r\n",
	))

	// Mark start time for prefill timing
	startTime := time.Now()

	// Write message_start IMMEDIATELY before connecting to vLLM
	eventData, _ := json.Marshal(map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id": "msg_01", "type": "message", "role": "assistant",
			"model": vllmModel, "content": []interface{}{},
			"stop_reason": nil, "stop_sequence": nil,
		},
		"usage": map[string]interface{}{"input_tokens": 0},
	})
	conn.Write([]byte("data: " + string(eventData) + "\n\n"))

	// Discard buffered data from the hijack without blocking
	if bufrw != nil {
		bufrw.Reader.Discard(bufrw.Reader.Buffered())
	}

	outputTokens := streamVLLM(ctx, conn, openaiJSON, startTime)

	// Wait briefly to ensure message_stop is fully received before closing
	time.Sleep(200 * time.Millisecond)
	return outputTokens, 0 // input tokens sent in message_delta, not available here
}

func streamVLLM(ctx context.Context, w sseWriter, openaiJSON []byte, startTime time.Time) int {
	_, span := otelTracer.Start(ctx, "proxy.stream")
	defer span.End()

	log.Println("[STREAM] Starting vLLM request")
	resp, err := requestVLLM(ctx, openaiJSON)
	if err != nil {
		log.Printf("[STREAM] vLLM error: %v", err)
		span.SetStatus(codes.Error, "vLLM error")
		s := newStreamState("msg_01", 0, startTime)
		finishStream(w, s, 0, "error")
		return 0
	}
	log.Println("[STREAM] vLLM response received")
	defer resp.Body.Close()

	s := newStreamState("msg_01", 0, startTime)
	totalTokens := 0
	totalChunks := 0

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		tokensAdded, stopReason, done := processVLLMLine(line, s, w)
		totalChunks++
		if done {
			totalTokens += tokensAdded
			finishStream(w, s, totalTokens, stopReason)
			span.SetAttributes(
				attribute.Int("llm.output.tokens", totalTokens),
				attribute.Int("proxy.stream.chunks", totalChunks),
			)
			return totalTokens
		}
		totalTokens += tokensAdded
	}
	stopReason := "end_turn"
	if len(s.tcs) > 0 {
		stopReason = "tool_use"
	}
	log.Printf("[STREAM] Done, tokens=%d, chunks=%d, stopReason=%s", totalTokens, totalChunks, stopReason)
	finishStream(w, s, totalTokens, stopReason)
	span.SetAttributes(
		attribute.Int("llm.output.tokens", totalTokens),
		attribute.Int("proxy.stream.chunks", totalChunks),
	)
	return totalTokens
}

// ─── Fallback streaming ───────────────────────────────────────────────────

func handleStreamingFallback(w http.ResponseWriter, ctx context.Context, openaiJSON []byte) (int, int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return 0, 0
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "close")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	eventData, _ := json.Marshal(map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id": "msg_01", "type": "message", "role": "assistant",
			"model": vllmModel, "content": []interface{}{},
			"stop_reason": nil, "stop_sequence": nil,
		},
		"usage": map[string]interface{}{"input_tokens": 0},
	})
	w.Write([]byte("data: " + string(eventData) + "\n\n"))
	flusher.Flush()

	// Wrap ResponseWriter + flusher into an sseWriter that auto-flushes
	flushingWriter := &flushingRW{w: w, f: flusher}
	return streamVLLM(ctx, flushingWriter, openaiJSON, time.Now()), 0
}

type flushingRW struct {
	w http.ResponseWriter
	f http.Flusher
}

func (f *flushingRW) Write(p []byte) (int, error) {
	n, err := f.w.Write(p)
	f.f.Flush()
	return n, err
}

// ─── HTTP handlers ────────────────────────────────────────────────────────

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"data": []map[string]interface{}{{"id": "qwen3.6-27b", "name": "qwen3.6-27b"}},
	})
}

func handleMessages(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	// Extract incoming trace context for end-to-end traces
	ctx := r.Context()
	if r.Header.Get("traceparent") != "" {
		ctx = otel.GetTextMapPropagator().Extract(ctx, propagation.HeaderCarrier(r.Header))
	}

	_, span := otelTracer.Start(ctx, "proxy.request")
	defer span.End()

	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		span.SetStatus(codes.Error, "method not allowed")
		otelRecordEnd(ctx, time.Since(startTime).Milliseconds(), "", false, "error", 0, 0, 0, 0)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		span.SetStatus(codes.Error, "read error")
		otelRecordEnd(ctx, time.Since(startTime).Milliseconds(), "", false, "error", 0, 0, 0, 0)
		http.Error(w, fmt.Sprintf("Read error: %v", err), http.StatusBadRequest)
		return
	}

	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		span.SetStatus(codes.Error, "invalid request")
		otelRecordEnd(ctx, time.Since(startTime).Milliseconds(), "", false, "error", 0, 0, 0, 0)
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	log.Printf("[REQ] model=%s stream=%v messages=%d tools=%v system=%v", req.Model, req.Stream, len(req.Messages), len(req.Tools), req.System != nil)

	// Reject Claude Code's internal requests (e.g. claude-haiku for reflection/eval)
	if strings.HasPrefix(req.Model, "claude-") {
		log.Printf("[REQ] unsupported claude model %q, returning 404", req.Model)
		span.SetStatus(codes.Error, "unsupported model")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, `{"type":"error","error":{"type":"not_found_error","message":"model %q not found"}}`, req.Model)
		otelRecordEnd(ctx, time.Since(startTime).Milliseconds(), req.Model, req.Stream, "unsupported_model", 0, 0, 0, 0)
		return
	}

	for i, m := range req.Messages {
		log.Printf("  [REQ] msg[%d] role=%s", i, m.Role)
	}
	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			var fn map[string]interface{}
			if err := json.Unmarshal(t, &fn); err == nil {
				log.Printf("  [REQ] tool: %v", fn["name"])
			}
		}
	}

	openaiReq := anthropicToOpenAI(ctx, &req)
	openaiJSON, err := json.Marshal(openaiReq)
	if err != nil {
		span.SetStatus(codes.Error, "marshal error")
		otelRecordEnd(ctx, time.Since(startTime).Milliseconds(), vllmModel, req.Stream, "error", 0, 0, 0, 0)
		http.Error(w, fmt.Sprintf("Marshal error: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("[REQ] OpenAI payload: %d bytes, tools=%v", len(openaiJSON), len(openaiReq.Tools))

	// Record request attributes on span
	span.SetAttributes(
		attribute.String("llm.request.model", req.Model),
		attribute.String("llm.request.model_mapped", vllmModel),
		attribute.Int("llm.request.messages", len(req.Messages)),
		attribute.Int("llm.request.tools", len(req.Tools)),
		attribute.Bool("llm.request.stream", req.Stream),
	)

	var outputTokens, inputTokens int
	if req.Stream {
		outputTokens, inputTokens = handleStreaming(w, ctx, openaiJSON)
	} else {
		outputTokens, inputTokens = handleCollected(ctx, w, openaiJSON)
	}

	otelRecordEnd(ctx, time.Since(startTime).Milliseconds(), vllmModel, req.Stream, "success", outputTokens, inputTokens, 0, 0)
}

// ─── Main ─────────────────────────────────────────────────────────────────

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := otelInit(ctx); err != nil {
		log.Printf("[OTEL] Init failed (continuing without telemetry): %v", err)
		otelTracer = otel.Tracer("golangproxy") // noop tracer when no provider is set
	}
	defer otelShutdown(context.Background())

	listenHost := os.Getenv("PROXY_HOST")
	listenPort := os.Getenv("PROXY_PORT")
	if listenHost == "" {
		listenHost = "0.0.0.0"
	}
	if listenPort == "" {
		listenPort = "4000"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", handleMessages)
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.Error(w, "Not found", http.StatusNotFound)
	})

	server := &http.Server{
		Addr:        fmt.Sprintf("%s:%s", listenHost, listenPort),
		Handler:     mux,
		ReadTimeout: 300 * time.Second,
		IdleTimeout: 60 * time.Second,
	}

	log.SetFlags(0)
	log.Printf("Anthropic→OpenAI Go proxy on %s:%s", listenHost, listenPort)
	log.Printf("  vLLM: %s  Model: %s  [Thinking ✓  Tools ✓  Streaming ✓  OTel ✓]", vllmURL, vllmModel)

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("[PROXY] Shutting down")
		shutdownCtx, release := context.WithTimeout(context.Background(), 30*time.Second)
		defer release()
		server.Shutdown(shutdownCtx)
	}()

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("[PROXY] Error: %v", err)
	}
}
