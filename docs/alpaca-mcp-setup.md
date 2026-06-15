# Alpaca MCP Operator Guide

## Setup
```bash
pip install alpaca-mcp-server
export ALPACA_API_KEY="key"
export ALPACA_SECRET_KEY="secret"
export ALPACA_PAPER_TRADE="true"
python -m alpaca_mcp_server
```

Claude.ai → Settings → Integrations → Add: `http://localhost:3001`

## What to ask Claude
- "What positions are open right now?"
- "What is my P&L today?"
- "Show all open orders"
- "What is my account equity?"

## Bot health check
`GET https://your-app.railway.app/health`

Returns: `status`, `last_scan`, `open_positions`, `daily_pnl`, `uptime`

Set as Railway health check path, port 8080. Override port with `HEALTH_PORT` env var.
