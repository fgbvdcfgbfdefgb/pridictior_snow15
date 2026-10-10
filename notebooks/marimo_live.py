"""CMC-inspired Binance BTC/USDT live chart for Marimo.

Run from the repository root with:
  marimo run notebooks/marimo_live.py --host 0.0.0.0 --port 2718

The refresh cell returns one Plotly figure. Marimo replaces that cell's output on
refresh rather than appending a new chart for every update.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from bitcoin_predictor.realtime import BinanceAggTradeFeed, LiveModelRunner

    return BinanceAggTradeFeed, LiveModelRunner, Path, go, make_subplots, mo, os, pd


@app.cell
def _(BinanceAggTradeFeed, os):
    symbol = os.getenv("BTC_SYMBOL", "BTCUSDT")
    seed_seconds = int(os.getenv("BTC_SEED_SECONDS", "86400"))
    feed = BinanceAggTradeFeed(symbol=symbol, seed_seconds=seed_seconds, seed_history=True).start()
    return (feed,)


@app.cell
def _(LiveModelRunner, feed, os):
    checkpoint_path = os.getenv("BTC_MODEL_PATH", "models/best.pt")
    runner = LiveModelRunner(checkpoint_path=checkpoint_path)
    # Seed the rolling feature engine once. Historical seed bars do not create
    # predictions; the first live prediction arrives on the next closed second.
    runner.warm_start(feed.seed_bars)
    return (runner,)


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["1s", "2s", "5s"], default_interval="1s")
    lookback = mo.ui.dropdown(options=["1h", "4h", "12h", "24h"], value="12h", label="Chart range")
    online_toggle = mo.ui.checkbox(value=False, label="Enable delayed online fine-tuning")
    controls = mo.hstack([lookback, online_toggle, refresh], justify="start", gap=1)
    return controls, lookback, online_toggle, refresh


@app.cell
def _(controls, feed, go, lookback, make_subplots, mo, online_toggle, pd, refresh, runner):
    # Referencing refresh.value makes only this output cell rerun. The feed and
    # runner live in upstream cells, so a refresh never starts another feed.
    _ = refresh.value
    runner.set_online(online_toggle.value)
    runner.process_bars(feed.bars_since(runner.last_timestamp))
    frame = feed.snapshot_frame()
    if frame.empty:
        status = feed.status
        if runner.load_error:
            status += f" · checkpoint load error: {runner.load_error}"
        rendered = mo.vstack([controls, mo.md(f"## BTC/USDT Spot\n\nWaiting for bars. {status}")])
        return (rendered,)

    seconds_by_range = {"1h": 3600, "4h": 4 * 3600, "12h": 12 * 3600, "24h": 24 * 3600}
    display_seconds = seconds_by_range.get(lookback.value, 12 * 3600)
    raw_view = frame.tail(display_seconds).copy()
    # The predictor still consumes each closed 1-second bar. Aggregate only the
    # browser chart to 1-minute candles so a 12/24-hour range stays responsive.
    view = (
        raw_view.set_index("timestamp")
        .resample("1min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    last = frame.iloc[-1]
    previous = frame.iloc[max(0, len(frame) - 60)]
    change_pct = (float(last["close"]) / max(float(previous["close"]), 1e-12) - 1.0) * 100.0
    change_color = "#16c784" if change_pct >= 0 else "#ea3943"
    forecast = runner.latest
    if forecast is None:
        forecast_text = "Model warming up (requires a complete 12-hour context)"
    else:
        forecast_text = (
            f"25-minute forecast: **${forecast['forecast_price']:,.2f}** "
            f"from ${forecast['current_price']:,.2f}"
        )
    status = feed.status
    if runner.model is None:
        status += " · no trained checkpoint loaded; showing live market data only"
    if runner.load_error:
        status += f" · checkpoint load error: {runner.load_error}"
    if runner.online_enabled:
        status += f" · delayed online updates: {runner._online_updates:,}"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.76, 0.24],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=view["timestamp"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="BTC/USDT",
            increasing_line_color="#16c784",
            decreasing_line_color="#ea3943",
            increasing_fillcolor="#16c784",
            decreasing_fillcolor="#ea3943",
        ),
        row=1,
        col=1,
    )
    if forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=[forecast["timestamp"], forecast["forecast_timestamp"]],
                y=[forecast["current_price"], forecast["forecast_price"]],
                mode="lines+markers",
                name="25m model forecast",
                line={"color": "#f0b90b", "width": 2, "dash": "dash"},
                marker={"size": 6},
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S UTC}<br>$%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    volume_colors = ["#16c784" if c >= o else "#ea3943" for o, c in zip(view["open"], view["close"])]
    fig.add_trace(
        go.Bar(x=view["timestamp"], y=view["volume"], marker_color=volume_colors, name="Volume", opacity=0.75),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        height=690,
        margin={"l": 48, "r": 20, "t": 24, "b": 30},
        paper_bgcolor="#0b0e11",
        plot_bgcolor="#0b0e11",
        font={"color": "#d1d4dc", "family": "Inter, Arial, sans-serif", "size": 12},
        legend={"orientation": "h", "y": 1.02, "x": 0.01},
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        uirevision="btc-live-chart",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#202630", linecolor="#323a46", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="#202630", linecolor="#323a46", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(showgrid=False, linecolor="#323a46", title_text="BTC volume", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    header = mo.md(
        f"""# BTC/USDT · Spot
**${float(last['close']):,.2f}**  
<span style='color:{change_color}'>1-minute move: {change_pct:+.3f}%</span>  
{forecast_text}  
<small>{status} · updated {pd.Timestamp(last['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}</small>"""
    )
    # Returning one Figure in this reactive output cell updates the existing
    # chart region; no display loop or accumulating outputs are used.
    rendered = mo.vstack([controls, header, fig])
    return (rendered,)


if __name__ == "__main__":
    app.run()
