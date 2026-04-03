import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Procurement Investigation Dashboard",
    layout="wide"
)


@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    return pd.read_csv("data.csv")


def parse_ai_prediction(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "UNKNOWN"

    if isinstance(value, dict):
        labels = value.get("labels", [])
        return ", ".join(str(x) for x in labels) if labels else str(value)

    text = str(value).strip()

    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                labels = parsed.get("labels", [])
                if isinstance(labels, list) and labels:
                    return ", ".join(str(x) for x in labels)
            return text
        except Exception:
            return text

    return text


def parse_ai_explanation(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "No AI explanation available."

    text = str(value).strip()

    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ["text", "content", "response", "message"]:
                    if key in parsed:
                        return str(parsed[key])
        except Exception:
            pass

    return text


def normalize_ai_label(label: str) -> str:
    if not label:
        return "UNKNOWN"

    text = str(label).upper()

    if "HIGH_RISK" in text:
        return "HIGH_RISK"
    if "MEDIUM_RISK" in text:
        return "MEDIUM_RISK"
    if "LOW_RISK" in text:
        return "LOW_RISK"
    return text


def ai_priority_score(ai_label: str) -> int:
    label = normalize_ai_label(ai_label)
    if label == "HIGH_RISK":
        return 3
    if label == "MEDIUM_RISK":
        return 2
    if label == "LOW_RISK":
        return 1
    return 0


def rule_priority_score(rule_flag: str) -> int:
    if rule_flag == "DUPLICATE_INVOICE":
        return 3
    if rule_flag == "OVERBILLING_RISK":
        return 3
    if rule_flag == "OK":
        return 1
    return 0


def recommend_action(rule_flag: str, ai_label: str) -> str:
    ai_label = normalize_ai_label(ai_label)

    if rule_flag in ["DUPLICATE_INVOICE", "OVERBILLING_RISK"] and ai_label == "HIGH_RISK":
        return "Block payment and escalate to finance review"
    if rule_flag in ["DUPLICATE_INVOICE", "OVERBILLING_RISK"]:
        return "Send for manual investigation before approval"
    if rule_flag == "OK" and ai_label == "HIGH_RISK":
        return "Review supporting documents for hidden anomalies"
    if rule_flag == "OK" and ai_label == "MEDIUM_RISK":
        return "Approve with analyst review"
    return "Approve"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["AI_RISK_PREDICTION_LABEL"] = df["AI_RISK_PREDICTION"].apply(parse_ai_prediction)
    df["AI_RISK_PREDICTION_LABEL"] = df["AI_RISK_PREDICTION_LABEL"].apply(normalize_ai_label)
    df["AI_EXPLANATION_TEXT"] = df["AI_EXPLANATION"].apply(parse_ai_explanation)

    df["RULE_SCORE"] = df["FINAL_FLAG"].apply(rule_priority_score)
    df["AI_SCORE"] = df["AI_RISK_PREDICTION_LABEL"].apply(ai_priority_score)
    df["COMBINED_SCORE"] = df["RULE_SCORE"] + df["AI_SCORE"]

    df["RECOMMENDED_ACTION"] = df.apply(
        lambda row: recommend_action(row["FINAL_FLAG"], row["AI_RISK_PREDICTION_LABEL"]),
        axis=1
    )

    df["RULE_AI_ALIGNMENT"] = df.apply(
        lambda row: "REVIEW"
        if (
            (row["FINAL_FLAG"] == "OK" and row["AI_RISK_PREDICTION_LABEL"] in ["MEDIUM_RISK", "HIGH_RISK"])
            or (row["FINAL_FLAG"] in ["DUPLICATE_INVOICE", "OVERBILLING_RISK"] and row["AI_RISK_PREDICTION_LABEL"] == "LOW_RISK")
            or (row["FINAL_FLAG"] == "DUPLICATE_INVOICE" and row["AI_RISK_PREDICTION_LABEL"] == "MEDIUM_RISK")
            or (row["FINAL_FLAG"] == "OVERBILLING_RISK" and row["AI_RISK_PREDICTION_LABEL"] == "MEDIUM_RISK")
        )
        else "ALIGNED",
        axis=1
    )

    return df


def compute_vendor_risk(df: pd.DataFrame) -> pd.DataFrame:
    vendor_df = (
        df.groupby("EXTRACTED_VENDOR_NAME")
        .agg(
            TOTAL_INVOICES=("FILE_NAME", "count"),
            TOTAL_AMOUNT=("EXTRACTED_AMOUNT", "sum"),
            DUPLICATE_CASES=("FINAL_FLAG", lambda s: int((s == "DUPLICATE_INVOICE").sum())),
            OVERBILLING_CASES=("FINAL_FLAG", lambda s: int((s == "OVERBILLING_RISK").sum())),
            AI_HIGH_RISK_CASES=("AI_RISK_PREDICTION_LABEL", lambda s: int((s == "HIGH_RISK").sum())),
            AVG_COMBINED_SCORE=("COMBINED_SCORE", "mean"),
            TOTAL_COMBINED_SCORE=("COMBINED_SCORE", "sum"),
        )
        .reset_index()
    )

    vendor_df["VENDOR_RISK_SCORE"] = (
        vendor_df["AVG_COMBINED_SCORE"] / 6.0 * 100
    ).round(1)

    vendor_df["RISK_TIER"] = vendor_df["VENDOR_RISK_SCORE"].apply(
        lambda x: "HIGH" if x >= 70 else ("MEDIUM" if x >= 40 else "LOW")
    )

    return vendor_df.sort_values(
        by=["VENDOR_RISK_SCORE", "TOTAL_COMBINED_SCORE"],
        ascending=[False, False]
    )


def local_ai_assistant(question: str, df: pd.DataFrame) -> str:
    q = question.lower()

    if "riskiest vendor" in q or "vendor is riskiest" in q:
        vendor_risk = compute_vendor_risk(df)
        if vendor_risk.empty:
            return "No vendor data is available."
        top = vendor_risk.iloc[0]
        return (
            f"The riskiest vendor is {top['EXTRACTED_VENDOR_NAME']} "
            f"with a vendor risk score of {top['VENDOR_RISK_SCORE']}."
        )

    if "how many duplicate" in q:
        count = int((df["FINAL_FLAG"] == "DUPLICATE_INVOICE").sum())
        return f"There are {count} duplicate invoice cases in the current filtered dataset."

    if "how many overbilling" in q or "overbilling" in q:
        count = int((df["FINAL_FLAG"] == "OVERBILLING_RISK").sum())
        return f"There are {count} overbilling risk cases in the current filtered dataset."

    if "review first" in q or "what should finance review first" in q:
        priority_df = df.sort_values(by=["COMBINED_SCORE", "EXTRACTED_AMOUNT"], ascending=[False, False])
        if priority_df.empty:
            return "There are no cases available to review."
        top = priority_df.iloc[0]
        return (
            f"Finance should review {top['FILE_NAME']} first. "
            f"It is flagged as {top['FINAL_FLAG']} with AI prediction {top['AI_RISK_PREDICTION_LABEL']} "
            f"and recommended action: {top['RECOMMENDED_ACTION']}."
        )

    if "high-risk" in q and "ok by rules" in q:
        subset = df[
            (df["AI_RISK_PREDICTION_LABEL"] == "HIGH_RISK") &
            (df["FINAL_FLAG"] == "OK")
        ]
        if subset.empty:
            return "There are no AI high-risk cases currently marked OK by rules."
        files = ", ".join(subset["FILE_NAME"].astype(str).tolist()[:5])
        return f"The following AI high-risk cases are still marked OK by rules: {files}."

    return (
        "I can answer questions about risky vendors, duplicate invoices, overbilling cases, "
        "top priority reviews, and rule-versus-AI disagreements."
    )


def render_sidebar_filters(df: pd.DataFrame):
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Executive Dashboard",
            "Case Explorer",
            "Risk Analytics",
            "AI Predictions",
            "AI Actions",
        ]
    )

    st.sidebar.markdown("---")

    vendor = st.sidebar.selectbox(
        "Filter by Vendor",
        ["All"] + sorted(df["EXTRACTED_VENDOR_NAME"].dropna().unique().tolist())
    )

    rule_flag = st.sidebar.selectbox(
        "Filter by Rule Flag",
        ["All"] + sorted(df["FINAL_FLAG"].dropna().unique().tolist())
    )

    ai_flag = st.sidebar.selectbox(
        "Filter by AI Prediction",
        ["All"] + sorted(df["AI_RISK_PREDICTION_LABEL"].dropna().unique().tolist())
    )

    filtered = df.copy()

    if vendor != "All":
        filtered = filtered[filtered["EXTRACTED_VENDOR_NAME"] == vendor]

    if rule_flag != "All":
        filtered = filtered[filtered["FINAL_FLAG"] == rule_flag]

    if ai_flag != "All":
        filtered = filtered[filtered["AI_RISK_PREDICTION_LABEL"] == ai_flag]

    return page, filtered


def render_kpis(data: pd.DataFrame):
    total_invoices = len(data)
    duplicate_count = len(data[data["FINAL_FLAG"] == "DUPLICATE_INVOICE"])
    overbilling_count = len(data[data["FINAL_FLAG"] == "OVERBILLING_RISK"])
    clean_count = len(data[data["FINAL_FLAG"] == "OK"])
    high_ai_count = len(data[data["AI_RISK_PREDICTION_LABEL"] == "HIGH_RISK"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Invoices", total_invoices)
    c2.metric("Duplicate Invoices", duplicate_count)
    c3.metric("Overbilling Risks", overbilling_count)
    c4.metric("Clean Invoices", clean_count)
    c5.metric("AI High Risk", high_ai_count)


def render_executive_dashboard(data: pd.DataFrame):
    st.title("📊 Executive Dashboard")
    st.caption("Public demo version using a static exported dataset. Production version runs on Snowflake + Cortex AI.")

    render_kpis(data)

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Priority Review Queue")
        queue = data.sort_values(
            by=["COMBINED_SCORE", "EXTRACTED_AMOUNT"],
            ascending=[False, False]
        )[
            [
                "FILE_NAME",
                "EXTRACTED_VENDOR_NAME",
                "EXTRACTED_INVOICE_NUMBER",
                "EXTRACTED_AMOUNT",
                "FINAL_FLAG",
                "AI_RISK_PREDICTION_LABEL",
                "RECOMMENDED_ACTION",
            ]
        ]
        st.dataframe(queue, use_container_width=True)

    with right:
        st.subheader("Top Risky Vendors")
        vendor_risk = compute_vendor_risk(data)[
            ["EXTRACTED_VENDOR_NAME", "VENDOR_RISK_SCORE"]
        ].set_index("EXTRACTED_VENDOR_NAME")
        st.bar_chart(vendor_risk)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Rule Risk Mix")
        rule_counts = (
            data["FINAL_FLAG"]
            .value_counts()
            .rename_axis("FINAL_FLAG")
            .reset_index(name="COUNT")
            .set_index("FINAL_FLAG")
        )
        st.bar_chart(rule_counts)

    with c2:
        st.subheader("AI Risk Mix")
        ai_counts = (
            data["AI_RISK_PREDICTION_LABEL"]
            .value_counts()
            .rename_axis("AI_RISK_PREDICTION_LABEL")
            .reset_index(name="COUNT")
            .set_index("AI_RISK_PREDICTION_LABEL")
        )
        st.bar_chart(ai_counts)


def render_case_explorer(data: pd.DataFrame):
    st.title("📄 Case Explorer")

    if data.empty:
        st.info("No invoices match the current filters.")
        return

    st.subheader("Filtered Investigation Queue")
    st.dataframe(
        data[
            [
                "FILE_NAME",
                "EXTRACTED_VENDOR_NAME",
                "EXTRACTED_INVOICE_NUMBER",
                "EXTRACTED_PO_ID",
                "EXTRACTED_AMOUNT",
                "MATCH_STATUS",
                "FINAL_FLAG",
                "AI_RISK_PREDICTION_LABEL",
            ]
        ],
        use_container_width=True
    )

    st.markdown("---")

    selected_file = st.selectbox(
        "Choose Invoice Case",
        data["FILE_NAME"].astype(str).tolist()
    )

    selected_row = data[data["FILE_NAME"].astype(str) == str(selected_file)].iloc[0]

    left, right = st.columns(2)

    with left:
        st.write("**File Name:**", selected_row["FILE_NAME"])
        st.write("**Vendor:**", selected_row["EXTRACTED_VENDOR_NAME"])
        st.write("**Invoice Number:**", selected_row["EXTRACTED_INVOICE_NUMBER"])
        st.write("**PO ID:**", selected_row["EXTRACTED_PO_ID"])
        st.write("**Invoice Amount:**", f"${float(selected_row['EXTRACTED_AMOUNT']):,.2f}")

    with right:
        st.write("**Match Status:**", selected_row["MATCH_STATUS"])
        st.write("**Rule Flag:**", selected_row["FINAL_FLAG"])
        st.write("**AI Prediction:**", selected_row["AI_RISK_PREDICTION_LABEL"])
        st.write("**Combined Score:**", selected_row["COMBINED_SCORE"])
        st.write("**Recommended Action:**", selected_row["RECOMMENDED_ACTION"])

    st.markdown("### 🤖 AI Explanation")
    st.write(selected_row["AI_EXPLANATION_TEXT"])


def render_risk_analytics(data: pd.DataFrame):
    st.title("📈 Risk Analytics")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Rule Risk Distribution")
        st.bar_chart(data["FINAL_FLAG"].value_counts())

    with c2:
        st.subheader("AI Risk Distribution")
        st.bar_chart(data["AI_RISK_PREDICTION_LABEL"].value_counts())

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Vendor Distribution")
        st.bar_chart(data["EXTRACTED_VENDOR_NAME"].value_counts())

    with c4:
        st.subheader("Invoice Amount by File")
        amount_chart = data.set_index("FILE_NAME")[["EXTRACTED_AMOUNT"]]
        st.bar_chart(amount_chart)

    st.markdown("---")

    st.subheader("Vendor-Level Risk Score")
    vendor_risk = compute_vendor_risk(data)
    st.dataframe(vendor_risk, use_container_width=True)

    st.markdown("### Vendor Risk Score Chart")
    chart_df = vendor_risk.set_index("EXTRACTED_VENDOR_NAME")[["VENDOR_RISK_SCORE"]]
    st.bar_chart(chart_df)


def render_ai_predictions(data: pd.DataFrame):
    st.title("🤖 AI Predictions")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI High Risk", len(data[data["AI_RISK_PREDICTION_LABEL"] == "HIGH_RISK"]))
    c2.metric("AI Medium Risk", len(data[data["AI_RISK_PREDICTION_LABEL"] == "MEDIUM_RISK"]))
    c3.metric("AI Low Risk", len(data[data["AI_RISK_PREDICTION_LABEL"] == "LOW_RISK"]))

    st.markdown("---")

    st.subheader("Rule vs AI Comparison")
    comparison = data[
        [
            "FILE_NAME",
            "EXTRACTED_VENDOR_NAME",
            "EXTRACTED_INVOICE_NUMBER",
            "FINAL_FLAG",
            "AI_RISK_PREDICTION_LABEL",
            "RULE_AI_ALIGNMENT",
        ]
    ]
    st.dataframe(comparison, use_container_width=True)

    st.markdown("---")

    st.subheader("Cases Needing Human Review")
    review_cases = data[data["RULE_AI_ALIGNMENT"] == "REVIEW"][
        [
            "FILE_NAME",
            "EXTRACTED_VENDOR_NAME",
            "EXTRACTED_INVOICE_NUMBER",
            "FINAL_FLAG",
            "AI_RISK_PREDICTION_LABEL",
            "AI_EXPLANATION_TEXT",
        ]
    ]
    if review_cases.empty:
        st.success("No rule-vs-AI disagreements found in the current filter.")
    else:
        st.dataframe(review_cases, use_container_width=True)


def render_ai_actions(data: pd.DataFrame):
    st.title("🧭 AI Actions")

    priority_df = data.sort_values(
        by=["COMBINED_SCORE", "EXTRACTED_AMOUNT"],
        ascending=[False, False]
    )

    st.subheader("Recommended Action Queue")
    st.dataframe(
        priority_df[
            [
                "FILE_NAME",
                "EXTRACTED_VENDOR_NAME",
                "EXTRACTED_INVOICE_NUMBER",
                "FINAL_FLAG",
                "AI_RISK_PREDICTION_LABEL",
                "RECOMMENDED_ACTION",
            ]
        ],
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Ask the Demo AI Assistant")
    st.caption("This public version uses a local rules-based assistant. The private production version uses Snowflake Cortex AI.")

    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []

    for item in st.session_state.assistant_history:
        st.markdown(f"**You:** {item['question']}")
        st.markdown(f"**AI:** {item['answer']}")
        st.markdown("---")

    user_question = st.text_input("Ask about the currently filtered invoice data")

    if st.button("Ask AI"):
        if user_question.strip():
            answer = local_ai_assistant(user_question, priority_df)
            st.session_state.assistant_history.append(
                {"question": user_question, "answer": answer}
            )
            st.success("Answer generated.")
            st.markdown(f"**AI:** {answer}")


def main():
    df = load_data()
    df = build_features(df)

    page, filtered_df = render_sidebar_filters(df)

    if page == "Executive Dashboard":
        render_executive_dashboard(filtered_df)
    elif page == "Case Explorer":
        render_case_explorer(filtered_df)
    elif page == "Risk Analytics":
        render_risk_analytics(filtered_df)
    elif page == "AI Predictions":
        render_ai_predictions(filtered_df)
    elif page == "AI Actions":
        render_ai_actions(filtered_df)


if __name__ == "__main__":
    main()