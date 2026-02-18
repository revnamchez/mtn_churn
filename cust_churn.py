import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Load Production Assets (Model, Scaler, Encoder, Column Order)
@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_tuned.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("encoder.joblib")
    model_cols = joblib.load("model_columns.joblib")
    return model, scaler, encoder, model_cols

model, scaler, encoder, model_cols = load_assets()

# --- UI SETUP ---
st.set_page_config(page_title="MTN Churn Predictor", layout="wide")
st.title("🟡 MTN Customer Churn Analysis")
st.markdown("### Tuned XGBoost Production Model (77% Accuracy)")

# --- INPUT SECTION (SIDEBAR) ---
with st.sidebar:
    st.header("Input Customer Details")
    
    # Categorical Inputs
    state = st.selectbox("State", ["Kwara", "Abuja", "Sokoto", "Gombe", "Oyo", "Plateau", "Jigawa", "Imo", "Bauchi", "Ondo", "Kebbi", "Adamawa", "Yobe", "Anambra", "Cross River", "Kogi", "Osun", "Kano", "Benue", "Rivers", "Enugu", "Borno", "Edo", "Kaduna", "Abia", "Ekiti", "Bayelsa", "Delta", "Zamfara", "Akwa Ibom", "Nasarawa", "Taraba", "Niger", "Katsina", "Lagos"])
    device = st.selectbox("MTN Device", ["4G Router", "Mobile SIM Card", "5G Broadband Router", "Broadband MiFi"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    plan = st.selectbox("Subscription Plan", ["165GB Monthly Plan", "12.5GB Monthly Plan", "150GB FUP Monthly Unlimited", "1GB+1.5mins Daily Plan", "30GB Monthly Broadband Plan", "10GB+10mins Monthly Plan", "25GB Monthly Plan", "7GB Monthly Plan", "1.5TB Yearly Broadband Plan", "65GB Monthly Plan", "120GB Monthly Broadband Plan", "300GB FUP Monthly Unlimited", "60GB Monthly Broadband Plan", "500MB Daily Plan", "3.2GB 2-Day Plan", "20GB Monthly Plan", "2.5GB 2-Day Plan", "450GB 3-Month Broadband Plan", "200GB Monthly Broadband Plan", "1.5GB 2-Day Plan", "16.5GB+10mins Monthly Plan"])
    
    # Numeric Inputs
    age = st.slider("Age", 18, 100, 30)
    satisfaction = st.slider("Satisfaction Rate (1-5)", 1, 5, 3)
    tenure = st.number_input("Tenure (Months)", 0, 3600, 12)
    purchases = st.number_input("Number of Purchases", 0, 100, 5)
    revenue = st.number_input("Total Revenue (₦)", 0.0, 50000.0, 2500.0)
    data_usage = st.number_input("Data Usage (GB)", 0.0, 5000.0, 15.0)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Analyze Churn Risk"):
    # 1. Create raw dataframe from inputs
    raw_input = pd.DataFrame([{
        'age': age, 'state': state, 'mtn_device': device, 'gender': gender,
        'satisfaction_rate': satisfaction, 'customer_tenure_in_months': tenure,
        'subscription_plan': plan, 'number_of_times_purchased': purchases,
        'total_revenue': revenue, 'data_usage': data_usage
    }])

    # 2. Encode Categorical Columns
    cat_cols = ['state', 'mtn_device', 'gender', 'subscription_plan']
    encoded_array = encoder.transform(raw_input[cat_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))

    # 3. Scale Numeric Columns
    num_cols = ['age', 'satisfaction_rate', 'customer_tenure_in_months', 
                'number_of_times_purchased', 'total_revenue', 'data_usage']
    scaled_num_array = scaler.transform(raw_input[num_cols])
    scaled_num_df = pd.DataFrame(scaled_num_array, columns=num_cols)

    # 4. Combine & Align Columns (CRITICAL for XGBoost)
    final_input = pd.concat([scaled_num_df, encoded_df], axis=1)
    final_input = final_input.reindex(columns=model_cols, fill_value=0)

    # 5. Prediction with the 'Result A+' 0.45 Threshold
    prob = model.predict_proba(final_input)[:, 1][0]
    prediction = 1 if prob >= 0.45 else 0

    # --- RESULTS DISPLAY ---
    col_metric, col_chart = st.columns([1, 2])

    with col_metric:
        st.subheader("Risk Assessment")
        if prediction == 1:
            st.error(f"⚠️ HIGH CHURN RISK: {prob:.1%}")
            st.write("Target this customer for a retention offer immediately.")
        else:
            st.success(f"✅ LOW CHURN RISK: {prob:.1%}")
            st.write("Customer appears stable. Maintain standard engagement.")
        
        st.metric("Churn Probability", f"{prob:.1%}")

    with col_chart:
        # Radar Chart: Comparing Model Architectures
        metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        fig = go.Figure()
        
        # Profile for Tuned Random Forest (Result A)
        fig.add_trace(go.Scatterpolar(
            r=[0.41, 0.53, 0.46, 0.64, 0.41],
            theta=metrics + [metrics],
            fill='toself', name='Tuned Random Forest', line_color='blue'
        ))
        
        # Profile for Tuned XGBoost (Result A+ Production)
        fig.add_trace(go.Scatterpolar(
            r=[0.65, 0.46, 0.54, 0.77, 0.65],
            theta=metrics + [metrics],
            fill='toself', name='Tuned XGBoost (Current)', line_color='gold'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Production Model Performance Benchmarks",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Adjust the customer details in the sidebar and click 'Analyze Churn Risk' to begin.")