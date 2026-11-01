"""Streamlit interface for ticket classification and triage."""
from pathlib import Path
import json
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.priority import assign_priority

st.set_page_config(page_title="TicketIQ | Support Triage", page_icon="🎫", layout="wide")
st.title("🎫 TicketIQ — Support Ticket Intelligence")
st.caption("NLP-based ticket categorization with transparent priority recommendations")

metrics_path = ROOT / "reports" / "metrics.json"
model_path = ROOT / "models" / "category_classifier.joblib"
if not metrics_path.exists() or not model_path.exists():
    st.warning("The model and evaluation reports are not available yet.")
    st.code("pip install -r requirements.txt\npython src/train.py\nstreamlit run app.py", language="bash")
    st.stop()

import joblib
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
model = joblib.load(model_path)
page = st.sidebar.radio("Navigate", ["Overview", "Classify a ticket", "Model evaluation", "Triage policy"])

if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training tickets", f"{metrics['train_rows']:,}")
    c2.metric("Test tickets", f"{metrics['test_rows']:,}")
    c3.metric("Issue categories", metrics["category_count"])
    c4.metric("Selected model", metrics["best_model"])
    st.write("The system converts ticket text into TF-IDF features, predicts one of seven IT support categories, and applies an explainable High / Medium / Low triage policy.")
    st.info("Priority is a rule-based recommendation because the source dataset contains no ground-truth priority labels. A support agent must review urgency.")
    chart = ROOT / "reports" / "category_distribution.png"
    if chart.exists():
        st.image(str(chart), caption="Ticket category distribution", use_container_width=True)

elif page == "Classify a ticket":
    st.subheader("Classify a new support ticket")
    ticket = st.text_area("Ticket subject and description", height=180, placeholder="Production server is down and multiple users cannot access the application...")
    if st.button("Analyze ticket", type="primary", disabled=not ticket.strip()):
        category = str(model.predict([ticket])[0])
        triage = assign_priority(ticket)
        a, b = st.columns(2)
        a.metric("Predicted category", category)
        b.metric("Suggested priority", triage["priority"])
        st.write("**Why this priority?**", triage["reason"])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([ticket])[0]
            classes = model.named_steps["classifier"].classes_
            st.write("**Category model probabilities**")
            st.bar_chart(pd.Series(probabilities, index=classes).sort_values(ascending=False))
        else:
            st.caption("The selected linear SVM does not expose calibrated class probabilities.")
        st.warning("Decision-support suggestion only; this is not an automatic dispatch or SLA guarantee.")

elif page == "Model evaluation":
    st.subheader("Held-out test performance")
    results = pd.DataFrame(metrics["models"]).T.sort_values("f1_macro", ascending=False)
    st.dataframe(results.style.format("{:.3f}"), use_container_width=True)
    st.caption(f"Best model selected by macro F1: {metrics['best_model']}. The supplied Zenodo test split is kept separate from training.")
    for filename, title in [("model_comparison.png", "Macro F1 comparison"), ("category_confusion_matrix.png", "Confusion matrix")]:
        path = ROOT / "reports" / filename
        if path.exists():
            st.image(str(path), caption=title, use_container_width=True)
    report_path = ROOT / "reports" / "classification_report.json"
    if report_path.exists():
        with st.expander("View class-wise precision, recall and F1"):
            st.json(json.loads(report_path.read_text(encoding="utf-8")))

else:
    st.subheader("Priority assignment policy")
    st.write("Priority is not learned from the dataset. It is assigned using a transparent keyword policy because the dataset does not contain verified urgency labels.")
    st.markdown("| Priority | Typical signals | Suggested handling |\n|---|---|---|\n| **High** | Service outage, production/server down, broad user impact, security incident | Immediate human review and escalation |\n| **Medium** | Operational errors, access problems, installation/configuration requests, unclear urgency | Standard support queue; confirm impact |\n| **Low** | How-to questions, documentation requests, suggestions, general inquiries | Routine queue |")
    st.warning("Rules are illustrative and must be adapted to an organization's incident policy. Ambiguous tickets default to Medium; human review remains mandatory.")
