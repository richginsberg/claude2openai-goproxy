package main

import (
	"context"
	"log"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

var (
	otelTracer      trace.Tracer
	otelMeter       metric.Meter
	reqDuration     metric.Int64Histogram
	reqCount        metric.Int64Counter
	reqTokens       metric.Int64Counter
	streamChunks    metric.Int64Counter
	tracerProvider  *sdktrace.TracerProvider
	meterProvider   *sdkmetric.MeterProvider
	otelEndpoint    string
)

func initEndpoint() string {
	e := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if e == "" {
		return "localhost:4317"
	}
	return e
}

func otelInit(ctx context.Context) error {
	otelEndpoint = initEndpoint()

	// Resource identifies this service
	res, err := resource.New(ctx,
		resource.WithSchemaURL(semconv.SchemaURL),
		resource.WithProcess(),
		resource.WithHost(),
		resource.WithAttributes(
			semconv.ServiceName("golangproxy"),
			semconv.ServiceVersion("1.0.0"),
		),
	)
	if err != nil {
		return err
	}

	// ── Tracer Provider ─────────────────────────────────────────────
	traceExp, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(otelEndpoint),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		return err
	}

	tracerProvider = sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(traceExp),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)
	otel.SetTracerProvider(tracerProvider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))
	otelTracer = tracerProvider.Tracer("golangproxy")

	// ── Meter Provider ──────────────────────────────────────────────
	metricExp, err := otlpmetricgrpc.New(ctx,
		otlpmetricgrpc.WithEndpoint(otelEndpoint),
		otlpmetricgrpc.WithInsecure(),
	)
	if err != nil {
		return err
	}

	meterProvider = sdkmetric.NewMeterProvider(
		sdkmetric.WithResource(res),
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExp,
			sdkmetric.WithInterval(10*time.Second),
		)),
	)
	otel.SetMeterProvider(meterProvider)
	otelMeter = meterProvider.Meter("golangproxy")

	// ── Instruments ─────────────────────────────────────────────────
	reqDuration, _ = otelMeter.Int64Histogram("proxy.request.duration",
		metric.WithDescription("End-to-end proxy request duration"),
		metric.WithUnit("ms"),
		metric.WithExplicitBucketBoundaries(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000),
	)

	reqCount, _ = otelMeter.Int64Counter("proxy.request.count",
		metric.WithDescription("Total proxy request count by status"),
	)

	reqTokens, _ = otelMeter.Int64Counter("proxy.request.tokens",
		metric.WithDescription("Token counts (input/output) per request"),
	)

	streamChunks, _ = otelMeter.Int64Counter("proxy.stream.chunks",
		metric.WithDescription("SSE content chunks written during streaming"),
	)

	log.Printf("[OTEL] Tracing → %s (gRPC)  |  Service: golangproxy", otelEndpoint)
	return nil
}

func otelShutdown(ctx context.Context) {
	if meterProvider != nil {
		if err := meterProvider.Shutdown(ctx); err != nil {
			log.Printf("[OTEL] meter shutdown error: %v", err)
		}
	}
	if tracerProvider != nil {
		if err := tracerProvider.Shutdown(ctx); err != nil {
			log.Printf("[OTEL] tracer shutdown error: %v", err)
		}
	}
}

// otelRecordEnd records request-level metrics at the end of handleMessages.
func otelRecordEnd(ctx context.Context, durationMs int64, model string, streaming bool, status string, outputTokens, inputTokens int, cacheReadTokens, cacheWriteTokens int) {
	if reqDuration == nil {
		return
	}
	attrs := []attribute.KeyValue{
		attribute.String("llm.request.model", model),
		attribute.Bool("llm.request.stream", streaming),
		attribute.String("proxy.request.status", status),
		attribute.String("user.name", proxyUser),
	}

	reqDuration.Record(ctx, durationMs, metric.WithAttributes(attrs...))
	reqCount.Add(ctx, 1, metric.WithAttributes(attrs...))

	if outputTokens > 0 {
		reqTokens.Add(ctx, int64(outputTokens), metric.WithAttributes(
			attribute.String("llm.request.model", model),
			attribute.String("llm.io", "output"),
		))
	}
	if inputTokens > 0 {
		reqTokens.Add(ctx, int64(inputTokens), metric.WithAttributes(
			attribute.String("llm.request.model", model),
			attribute.String("llm.io", "input"),
		))
	}
	if cacheReadTokens > 0 {
		reqTokens.Add(ctx, int64(cacheReadTokens), metric.WithAttributes(
			attribute.String("llm.request.model", model),
			attribute.String("llm.io", "cache_read"),
		))
	}
	if cacheWriteTokens > 0 {
		reqTokens.Add(ctx, int64(cacheWriteTokens), metric.WithAttributes(
			attribute.String("llm.request.model", model),
			attribute.String("llm.io", "cache_write"),
		))
	}
}
