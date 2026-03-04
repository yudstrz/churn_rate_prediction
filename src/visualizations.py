import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def style_dataframe(df):
    """Apply conditional formatting to the prediction dataframe."""
    def highlight_row(row):
        color = 'background-color: rgba(255, 75, 75, 0.1)' if row.get('High Risk', 0) == 1 else ''
        return [color] * len(row)
    return df.style.apply(highlight_row, axis=1)

def plot_density_chart(df, threshold):
    """Render a seaborn density plot for churn probability distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Churn Probability"], bins=20, kde=True, ax=ax, color="#1f77b4")
    ax.axvline(threshold, color="#d62728", linestyle="--", label=f"High Risk Threshold ({threshold})")
    ax.set_title("Distribution (Density)", fontsize=12)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.legend()
    return fig

def plot_interactive_histogram(df, threshold):
    """Render an interactive plotly histogram for churn probability."""
    fig_plotly = px.histogram(
        df,
        x="Churn Probability",
        nbins=20,
        title="Distribution (Interactive)",
        color_discrete_sequence=["#1f77b4"]
    )
    fig_plotly.add_vline(x=threshold, line_dash="dash", line_color="#d62728", annotation_text="Threshold ")
    fig_plotly.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig_plotly
