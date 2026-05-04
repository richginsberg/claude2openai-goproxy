# golangproxy

A high-performance Go proxy that translates the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) to the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat). Designed to run between Claude Code (or any Anthropic SDK client) and a local [vLLM](https://github.com/vllm-project/vllm) server, enabling you to use open-weight models through tools that expect the Anthropic API.

## Features

- **Full streaming support** — SSE events are translated in real-time with zero buffering via raw socket hijacking
- **Tool use** — Converts Anthropic tool definitions (`{"name": ..., "input_schema": ...}`) and OpenAI function calls bidirectionally; partial JSON is streamed as `input_json_delta` events
- **Thinking / reasoning** — vLLM reasoning content is mapped to Anthropic's `thinking_delta` event type
- **Non-streaming fallback** — Collected mode assembles the full response before returning
- **Dual format support** — `convertTools` handles both Anthropic format (`{"name": "...", "input_schema": {...}}`) and OpenAI format (`{"type": "function", "function": {...}}`)
- **Clean stream completion** — Sends `message_stop` at the end of streaming responses per the Anthropic API spec
- **Reject unsupported models** — Immediately returns 404 for unhandled model IDs (e.g. Claude Code's internal evaluation requests), preventing queue buildup at vLLM
- **Guarded tool call parsing** — Ignores empty `id`/`name` in trailing vLLM chunks to prevent overwriting valid values
- **Client timeout** — 120-second timeout on all vLLM requests to prevent indefinite hangs
- **Threaded request handling** — Each connection is handled in its own goroutine
- **OpenTelemetry** — Traces and metrics exported via OTLP gRPC; supports end-to-end tracing with `traceparent` propagation through to vLLM
- **OpenTelemetry observability** — Traces and metrics exported via OTLP (spans: `proxy.request`, `proxy.translate.request`, `proxy.forward.vllm`, `proxy.stream`; metrics: duration, count, tokens, chunks)

## Architecture

```
┌──────────────┐       Anthropic API        ┌──────────────┐    OpenAI Chat     ┌─────────┐
│  Claude Code  │ ──────────────────────▶    │  golangproxy  │ ────────────────▶ │   vLLM  │
│  (or any     │  /v1/messages, SSE stream   │  :4000        │  /v1/chat/comp    │  :6006  │
│  Anthropic    │ ◀────────────────────────  │  (Go)         │  letions, SSE     │  (LLM)  │
│  SDK client)  │      Anthropic SSE events   │               │                   │         │
└──────────────┘                             └──────────────┘                   └─────────┘
```

The proxy intercepts Anthropic-format requests, translates them to OpenAI format, forwards them to vLLM, then translates the streaming SSE response back to Anthropic format in real-time.

## Prerequisites

- [Go 1.23+](https://go.dev/dl/)
- A running [vLLM](https://github.com/vllm-project/vllm) server (default: `http://localhost:6006`)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd golangproxy

# Build the binary
go build -o golangproxy main.go
```

## Usage

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_URL` | `http://localhost:6006` | Base URL of the vLLM server |
| `VLLM_MODEL` | `Lorbus/Qwen3.6-27B-int4-AutoRound` | Model ID to use in requests to vLLM |
| `PROXY_PORT` | `4000` | Port the proxy listens on |
| `PROXY_HOST` | `0.0.0.0` | Host the proxy binds to |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `localhost:4317` | OTLP gRPC endpoint for traces/metrics |

### Running Manually

```bash
./golangproxy

# With custom settings
VLLM_URL=http://localhost:8080 VLLM_MODEL=meta-llama/Llama-3.1-8B-instruct PROXY_PORT=5000 ./golangproxy
```

### Running with systemd

An install script configures the systemd user service with your settings:

```bash
# Install with defaults (binary → /usr/local/bin, vLLM → localhost:6006)
./install-systemd.sh

# Or customize with environment variables
VLLM_URL=http://localhost:8080 VLLM_MODEL=meta-llama/Llama-3.1-8B-instruct PROXY_PORT=5000 ./install-systemd.sh
```

The script builds the binary (if needed), copies it to `/usr/local/bin`, and generates a service file from the template at `anthropic-proxy.service`.

```bash
# Check status
systemctl --user status anthropic-proxy.service

# View logs
journalctl --user -u anthropic-proxy.service -f

# Stop and disable
systemctl --user disable --now anthropic-proxy.service
```

For a custom install path, set `BIN_DIR`:

```bash
BIN_DIR=/opt/bin VLLM_MODEL=my-org/my-model ./install-systemd.sh
```

## Configuration with Claude Code

Point Claude Code to the proxy:

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-local  # any non-empty value works
claude
```

The proxy presents a compatible `/v1/messages` endpoint and `/v1/models` listing, so Claude Code connects without modification.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Main messages endpoint — translates Anthropic ↔ OpenAI |
| `GET` | `/v1/models` | Returns a model list compatible with Anthropic clients |
| `HEAD` | `/` | Health check |

## Format Translation

### Requests (Anthropic → OpenAI)

- System prompts (string or block array) → `role: "system"` message
- Content blocks (`text`, `tool_use`, `tool_result`) → flat text / `tool_calls` / `role: "tool"`
- `stop_sequences` → `stop`
- `max_tokens`, `temperature`, `top_p`, `top_k` → passthrough
- Tools in Anthropic format → OpenAI function calling format

### Streaming Response (OpenAI → Anthropic)

| vLLM field | Anthropic event |
|---|---|
| `delta.content` | `content_block_delta` with `text_delta` |
| `delta.reasoning_content` / `delta.reasoning` | `content_block_delta` with `thinking_delta` |
| `delta.tool_calls[*].function.name` | `content_block_start` with `tool_use` |
| `delta.tool_calls[*].function.arguments` | `content_block_delta` with `input_json_delta` |
| `finish_reason` | `message_delta` → `message_stop` |

## Building

```bash
# Fetch dependencies, then build
go mod tidy
go build -o golangproxy main.go

# Cross-compile for Linux amd64
GOOS=linux GOARCH=amd64 go build -o golangproxy
```

## Observability (OpenTelemetry)

The proxy exports traces and metrics via OTLP gRPC to a local OpenTelemetry Collector on port `4317` by default.

### Traces

| Span | Description |
|---|---|
| `proxy.request` | Root span for the full request lifecycle |
| `proxy.translate.request` | Anthropic → OpenAI format translation |
| `proxy.forward.vllm` | Outbound request to vLLM (propagates `traceparent`) |
| `proxy.stream` | Streaming response processing (attributes: `llm.output.tokens`, `proxy.stream.chunks`) |
| `proxy.collect` | Non-streaming response processing (attributes: `llm.output.tokens`) |

End-to-end traces are supported: if Claude Code (or any client) sends a `traceparent` header, it is extracted and used as the parent context. The trace context is also propagated into the outbound vLLM request.

### Metrics

| Metric | Type | Attributes |
|---|---|---|
| `proxy.request.duration` | Histogram (ms) | `llm.request.model`, `llm.request.stream`, `proxy.request.status` |
| `proxy.request.count` | Counter | `llm.request.model`, `llm.request.stream`, `proxy.request.status` |
| `proxy.request.tokens` | Counter | `llm.request.model`, `llm.io` (`input`/`output`) |
| `proxy.stream.chunks` | Counter | recorded on span attributes |

### Running the Collector

```bash
# Start the collector with ClickHouse exporter
# Requires: otel/opentelemetry-collector-contrib (includes clickhouseexporter)
docker run -d --name otel-collector \
  -p 4317:4317 -p 4318:4318 \
  -v $(pwd)/otelcol-config.yml:/etc/otelcol/config.yml \
  otel/opentelemetry-collector-contrib:latest \
  --config /etc/otelcol/config.yml
```

A sample collector config is included at `otelcol-config.yml`. It exports to stdout (for debugging) and ClickHouse (`localhost:8123`, database `otel`). Adjust the ClickHouse endpoint in the config for your setup.

To disable OTel entirely (no collector available), start the proxy with an unreachable endpoint — the exporter retries briefly then drops data silently:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=127.0.0.1:1 ./golangproxy  # falls back gracefully
```

## License

MIT
