# Grafana Setup for golangproxy OpenTelemetry

## 1. Install the ClickHouse plugin

```bash
grafana-cli plugins install vertamedia-clickhouse-datasource
# Then restart Grafana:
docker restart grafana
```

## 2. Add ClickHouse data source

In Grafana: **Configuration → Data Sources → Add data source → ClickHouse**

| Setting | Value |
|---|---|
| **Name** | `otel-ch` (must match — the dashboard uses this UID) |
| **URL** | `otel-ch-otel:8124` (if Grafana is on the `otel-net` Docker network) |
| **Username** | `default` |
| **Password** | `otel` |
| **Database** | `otel` |
| **TLS** | None |

> **If Grafana runs on the host** (not in Docker on `otel-net`), use URL: `localhost:8124`

## 3. Import the dashboard

In Grafana: **New → Import → Upload JSON file** → select `grafana-dashboard.json`

The dashboard will auto-detect the `otel-ch` data source via the `ds-otel-ch` variable.

## Dashboard structure

| Row | Panels |
|---|---|
| **Overview** | Total Requests, Avg Duration, P95 Duration, Error Rate |
| **Duration Breakdown** | Avg Duration by Span (bargauge), Duration Distribution (barchart) |
| **Time Series** | Requests Over Time, Avg Duration Over Time |
| **Tokens** | Output Tokens Over Time, Avg Tokens/Request |
| **Trace Explorer** | Recent Spans table (200 rows, sortable) |

All panels respect Grafana's time range selector.
