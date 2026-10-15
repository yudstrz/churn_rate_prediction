import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def style_dataframe(df):
    def highlight_row(row):
        color = 'background-color: rgba(255, 75, 75, 0.1)' if row.get('High Risk', 0) == 1 else ''
        return [color] * len(row)
    return df.style.apply(highlight_row, axis=1)

def plot_density_chart(df, threshold):
    probs = df["Churn Probability"].dropna().values
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=probs,
        histnorm="probability density",
        name="Density",
        nbinsx=20,
        marker_color="rgba(37, 99, 235, 0.4)",
        marker_line_color="#2563EB",
        marker_line_width=1.5,
    ))

    if len(probs) > 1:
        x_range = np.linspace(0, 1, 200)
        std = np.std(probs) or 0.1
        h = max(1.06 * std * (len(probs) ** (-0.2)), 0.02)
        kde = np.sum(np.exp(-0.5 * ((x_range[:, None] - probs) / h) ** 2), axis=1) / (len(probs) * h * np.sqrt(2 * np.pi))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde,
            mode="lines",
            name="KDE",
            line=dict(color="#60A5FA", width=2.5)
        ))

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold})"
    )

    fig.update_layout(
        title="Distribution (Density)",
        xaxis_title="Churn Probability",
        yaxis_title="Density",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_interactive_histogram(df, threshold):
    fig_plotly = px.histogram(
        df,
        x="Churn Probability",
        nbins=20,
        title="Distribution (Count)",
        color_discrete_sequence=["#2563EB"]
    )
    fig_plotly.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold})"
    )
    fig_plotly.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig_plotly

def plot_feature_importance(model, features):
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(features):
        return None
    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Model Feature Importance (Key Drivers)",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    return fig

def plot_value_risk_scatter(df, threshold):
    charge_cols = [c for c in ["total day charge", "total eve charge", "total night charge", "total intl charge"] if c in df.columns]
    total_charges = df[charge_cols].sum(axis=1) if charge_cols else pd.Series(0, index=df.index)
    plot_df = df.copy()
    plot_df["Total Monthly Charges ($)"] = total_charges
    plot_df["Risk Category"] = plot_df["High Risk"].map({1: "High Risk", 0: "Low/Medium Risk"})

    median_charge = total_charges.median()

    fig = px.scatter(
        plot_df,
        x="Churn Probability",
        y="Total Monthly Charges ($)",
        color="Risk Category",
        color_discrete_map={"High Risk": "#EF4444", "Low/Medium Risk": "#3B82F6"},
        hover_data=["customer service calls", "total day minutes"] if "customer service calls" in df.columns else None,
        title="Customer Value vs Churn Risk Matrix"
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#EF4444", annotation_text="Risk Threshold")
    if median_charge > 0:
        fig.add_hline(y=median_charge, line_dash="dot", line_color="#94A3B8", annotation_text="Median Spend")

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig
