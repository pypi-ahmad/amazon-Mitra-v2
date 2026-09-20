"""Render locally generated synthetic-prior illustrations."""

import plotly.express as px
import streamlit as st

from src.config import PAPER_URL
from src.synthetic_gallery import synthetic_prior_shapes

st.html('<div class="gb-kicker">04 · Architecture</div>')
st.title("Inside Mitra-v2")
st.write(
    "Mitra-v2 learns broad table patterns from synthetic pretraining before a real table arrives."
)

st.metric("REAL TABLES SEEN", "0", border=True)
st.caption(
    "These locally generated illustrations show synthetic pattern families. They are "
    "not Amazon figures or samples from the model's training corpus."
)

gallery = synthetic_prior_shapes()
columns = st.columns(2)
for index, (title, frame) in enumerate(gallery.items()):
    with columns[index % 2], st.container(border=True):
        st.subheader(title)
        st.html('<span class="gb-generated">generated · not real</span>')
        figure = px.scatter(
            frame,
            x="x",
            y="y",
            color="pattern",
            color_discrete_sequence=["#9b87f5", "#46c2b3", "#f39a5a", "#d66eca"],
        )
        figure.update_traces(marker={"size": 5, "opacity": 0.78})
        figure.update_layout(
            height=300,
            margin={"l": 8, "r": 8, "t": 12, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0b1220",
            font={"color": "#dbe2ff"},
            showlegend=False,
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

st.subheader("Simplified view")
st.code("table → row/column tokens → 2D attention → target distribution", language=None)
st.caption(
    f"This simplified view follows the [technical report]({PAPER_URL}). It does not reproduce the paper's figures or establish causal structure."
)
