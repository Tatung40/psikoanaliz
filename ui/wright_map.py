"""Wright map (person–item map) figure shared by the Rasch screens. No estimation."""
import numpy as np
import plotly.graph_objects as go


def wright_map(result, bins=30):
    """Person WLE histogram (left) and item difficulties (right) on one logit axis."""
    persons = np.asarray(result["wright"]["persons"], float)
    items = result["wright"]["items"]
    lo = float(min(persons.min(), items.min())) - .5
    hi = float(max(persons.max(), items.max())) + .5
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(persons, edges)
    centres = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    fig.add_bar(x=-counts, y=centres, orientation="h", name="Kişiler (WLE)",
                marker_color="#4C78A8", width=(edges[1] - edges[0]) * .9,
                customdata=counts, hovertemplate="θ ≈ %{y:.2f}<br>kişi: %{customdata}<extra></extra>")
    fig.add_scatter(x=np.full(len(items), counts.max() * .15 + 1), y=items.to_numpy(float),
                    mode="markers+text", text=list(items.index), textposition="middle right",
                    name="Maddeler (b)", marker=dict(symbol="diamond", size=10, color="#E45756"),
                    hovertemplate="%{text}<br>b = %{y:.2f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_white", title="Wright Haritası (kişi–madde eşlemesi)",
                      xaxis=dict(title="← kişi sayısı | maddeler →", showticklabels=False, zeroline=True),
                      yaxis=dict(title="Logit (θ ve b)", range=[lo, hi]), barmode="overlay",
                      legend=dict(orientation="h", y=-.15), height=520, margin=dict(l=10, r=10, t=50, b=10))
    return fig
