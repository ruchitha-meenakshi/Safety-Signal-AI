import streamlit as st
import pandas as pd
import plotly.express as px
from databricks import sql

# ------------------------------------------------------------------
# 1. UI CONFIGURATION & STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaSafety AI", 
    layout="wide", 
    page_icon="💊",
    initial_sidebar_state="expanded"
)

# Enterprise-Grade CSS
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #464b5c;
    }
    /* Remove Plotly Chart Backgrounds */
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
    /* Cleaner Expander Headers */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. DATA CONNECTION
# ------------------------------------------------------------------
try:
    SERVER_HOSTNAME = st.secrets["DB_HOSTNAME"]
    HTTP_PATH = st.secrets["DB_HTTP_PATH"]
    ACCESS_TOKEN = st.secrets["DB_ACCESS_TOKEN"]
except Exception as e:
    st.error(f"❌ Missing Secrets! Check .streamlit/secrets.toml. Error: {e}")
    st.stop()

CATALOG = "safety_signal_catalog"
SCHEMA = "raw_data"

@st.cache_data(ttl=600)
def load_data():
    try:
        connection = sql.connect(
            server_hostname=SERVER_HOSTNAME,
            http_path=HTTP_PATH,
            access_token=ACCESS_TOKEN,
            _disable_cloud_fetch=True
        )
        query = f"SELECT * FROM {CATALOG}.{SCHEMA}.gold_model_predictions LIMIT 5000"
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            if not result: return pd.DataFrame()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(result, columns=columns)
            
            # Cleaning & Parsing
            if 'condition' in df.columns:
                df = df[~df['condition'].astype(str).str.contains("users found|</span>|<span", case=False, regex=True)]
            
            if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], errors='coerce')
            elif 'event_date' in df.columns: df['date'] = pd.to_datetime(df['event_date'], errors='coerce')
            
            def parse_risk_score(val):
                try:
                    if isinstance(val, str):
                        clean_val = val.replace('[', '').replace(']', '') 
                        parts = clean_val.split(',')
                        return float(parts[1]) if len(parts) > 1 else float(parts[0])
                    return float(val)
                except: return 0.0

            if 'probability' in df.columns:
                df['probability'] = df['probability'].apply(parse_risk_score)
        connection.close()
        return df
    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ------------------------------------------------------------------
with st.sidebar:
    st.title("💊 PharmaSafety AI")
    st.caption("Pharmacovigilance Suite v2.0")
    
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        The **PharmaSafety Monitor** assists safety teams in the rapid triage of adverse drug events (ADEs).
        
        **Workflow**
        * **Select** a target drug for surveillance.
        * **Analyze** risk stratification charts.
        * **Adjudicate** high-confidence signals in the review log.
        """)
    
    st.divider()
    
    with st.spinner('Syncing with Lakehouse...'):
        df = load_data()

    if not df.empty:
        st.subheader("🎛️ Parameters")
        
        # Drug Filter
        drug_list = ["All Drugs"] + sorted(df['drugName'].astype(str).unique().tolist())
        selected_drug = st.selectbox("Target Drug:", drug_list, help="Select active ingredient or brand name")

        # Condition Filter
        if selected_drug != "All Drugs":
            relevant_conditions = df[df['drugName'] == selected_drug]['condition'].astype(str).unique().tolist()
        else:
            relevant_conditions = df['condition'].astype(str).unique().tolist()
        
        condition_list = ["All Conditions"] + sorted(relevant_conditions)
        selected_condition = st.selectbox("Condition Focus:", condition_list, help="Filter by reported medical condition")

        # Date Filter
        date_range = None
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            min_date, max_date = df['date'].min(), df['date'].max()
            if pd.notnull(min_date) and pd.notnull(max_date):
                st.markdown("###") 
                date_range = st.date_input("📅 Reporting Period", [min_date, max_date])

        st.divider()
        show_all_reviews = st.toggle("Include Safe Cases", value=False)
        
        st.markdown("---")
        st.caption("Codebasics Resume Project Challenge • Built with Databricks & Streamlit by Ruchitha Uppuluri")

# ------------------------------------------------------------------
# 4. MAIN DASHBOARD
# ------------------------------------------------------------------
if not df.empty:
    # FILTER LOGIC
    filtered_df = df.copy()
    if selected_drug != "All Drugs": filtered_df = filtered_df[filtered_df['drugName'] == selected_drug]
    if selected_condition != "All Conditions": filtered_df = filtered_df[filtered_df['condition'] == selected_condition]
    if date_range and len(date_range) == 2 and 'date' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['date'].dt.date >= date_range[0]) & (filtered_df['date'].dt.date <= date_range[1])]

    # HEADER
    st.title(f"🛡️ Surveillance Dashboard: {selected_drug}")
    st.markdown("Real-time detection and risk assessment of patient safety signals.")
    st.markdown("###")

    # KPIS (WITH TOOLTIPS 💡)
    total = len(filtered_df)
    adverse = len(filtered_df[filtered_df['prediction'] == 1.0])
    rate = (adverse / total * 100) if total > 0 else 0
    avg_conf = filtered_df['probability'].mean() if not filtered_df.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        label="Total Narratives", 
        value=f"{total:,}", 
        help="Total number of patient reviews ingested and analyzed from the Lakehouse."
    )
    k2.metric(
        label="Detected Signals", 
        value=f"{adverse:,}", 
        delta_color="inverse", 
        help="Count of reviews flagged by the AI model as potential adverse drug events."
    )
    k3.metric(
        label="Signal Rate", 
        value=f"{rate:.1f}%", 
        help="Percentage of total reviews identified as safety signals (Signals / Total)."
    )
    k4.metric(
        label="Avg. Confidence", 
        value=f"{avg_conf:.2f}", 
        help="The model's average probability score across all displayed reviews (0.0 to 1.0)."
    )
    
    st.divider()

    # CHARTS
    c1, c2 = st.columns((2, 1))
    
    with c1:
        st.subheader("🚨 Top Reported Adverse Conditions")
        if not filtered_df.empty:
            top_risks = filtered_df[filtered_df['prediction'] == 1.0]['condition'].value_counts().head(10).reset_index()
            top_risks.columns = ['Condition', 'Count']
            
            fig = px.bar(top_risks, x='Count', y='Condition', orientation='h', 
                         color='Count', color_continuous_scale='Reds', text_auto=True)
            fig.update_layout(xaxis_title="Reported Cases", yaxis_title="", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("📊 Adverse Event Ratio")
        counts = filtered_df['prediction'].value_counts().reset_index()
        counts.columns = ['Pred', 'Count']
        counts['Label'] = counts['Pred'].map({1.0: 'Adverse Event', 0.0: 'Safe Experience'})
        
        fig2 = px.pie(counts, names='Label', values='Count', hole=0.6, 
                      color='Label', color_discrete_map={'Adverse Event':'#FF4B4B', 'Safe Experience':'#00CC96'})
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=True, legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

    # TABLE
    st.subheader("Individual Case Safety Report Listing")
    
    if show_all_reviews:
        display_df = filtered_df.sort_values(by='probability', ascending=False)
    else:
        display_df = filtered_df[filtered_df['prediction'] == 1.0].sort_values(by='probability', ascending=False)
        st.caption(f"Validation queue for {len(display_df)} high-confidence signals.")
    
    for index, row in display_df.head(20).iterrows():
        # Logic for Status Badge
        risk = row['probability']
        if row['prediction'] == 1.0:
            color = "🔴" if risk > 0.9 else "🟠"
            status = "Adverse Event"
        else:
            color = "🟢"
            status = "Safe"
        
        # Professional Label Format
        label = f"**{color} {row['condition']}** | {status} (Conf: {risk:.0%})"
        
        with st.expander(label):
            c1, c2 = st.columns((3, 1))
            with c1:
                st.markdown("**Patient Narrative**")
                st.markdown(f"> *\"{row['clean_review']}\"*")
            with c2:
                st.markdown("**Metadata**")
                st.caption(f"💊 **Drug:** {row['drugName']}")
                if 'date' in row and pd.notnull(row['date']):
                    st.caption(f"📅 **Date:** {row['date'].strftime('%Y-%m-%d')}")
                st.caption(f"⭐ **Rating:** {row.get('rating', 'N/A')}/10")

else:
    st.warning("⚠️ No data loaded. Check connection details.")

