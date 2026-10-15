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
