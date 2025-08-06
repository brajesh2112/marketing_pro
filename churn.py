import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Marketing Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Marketing Analytics Dashboard")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Churn_pred.csv") 
    except FileNotFoundError:
        st.error("❌ Churn_pred.csv file not found. Please upload the dataset.")
        return None
    
    # Store original data for reference
    df_original = df.copy()
    
    # Remove customer ID if exists
    df.drop(columns=['customerID'], errors='ignore', inplace=True)
    
    # Handle missing values
    df = df.dropna()
    
    # Convert TotalCharges to numeric (often stored as string)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = df.copy()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Keep original categorical values for interpretation
    df_with_originals = df.copy()
    
    # Scale numerical variables for clustering
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded, df_scaled, df_with_originals, scaler, label_encoders

# Load data
data_result = load_data()
if data_result is None:
    st.stop()

df_encoded, df_scaled, df_original, scaler, label_encoders = data_result

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis:",
    ["Customer Lifetime Value", "Customer Segmentation", "Market Basket Analysis", "A/B Testing", "Dashboard Overview"]
)

# --------------------------
# Customer Lifetime Value (CLV)
# --------------------------
if analysis_type == "Customer Lifetime Value" or analysis_type == "Dashboard Overview":
    st.header("💰 Customer Lifetime Value (CLV) Analysis")
    
    # Improved CLV calculation
    df_encoded['CLV'] = (df_encoded['MonthlyCharges'] * df_encoded['tenure'] * 
                        (1 - df_encoded.get('Churn', 0) * 0.5))  # Adjust for churn risk
    
    # Build CLV prediction model
    X_clv = df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y_clv = df_encoded['CLV']
    
    clv_model = LinearRegression()
    clv_model.fit(X_clv, y_clv)
    clv_r2 = r2_score(y_clv, clv_model.predict(X_clv))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 CLV Predictor")
        st.write(f"**Model Accuracy (R²): {clv_r2:.3f}**")
        
        # Input sliders with realistic ranges
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=1.0)
        total_charges = st.slider("Total Charges ($)", min_value=18.0, max_value=8500.0, value=float(tenure * monthly_charges), step=10.0)
        
        # Predict CLV
        input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                                  columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
        clv_prediction = clv_model.predict(input_data)[0]
        
        st.metric(label="💰 Estimated CLV", value=f"${clv_prediction:.2f}")
        
        # CLV interpretation
        if clv_prediction > df_encoded['CLV'].quantile(0.75):
            st.success("🌟 High Value Customer")
        elif clv_prediction > df_encoded['CLV'].quantile(0.25):
            st.info("📈 Medium Value Customer")
        else:
            st.warning("⚠️ Low Value Customer")
    
    with col2:
        st.subheader("📊 CLV Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_encoded['CLV'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(clv_prediction, color='red', linestyle='--', linewidth=2, label=f'Predicted CLV: ${clv_prediction:.2f}')
        ax.set_xlabel('Customer Lifetime Value ($)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('CLV Distribution')
        ax.legend()
        st.pyplot(fig)

# --------------------------
# Customer Segmentation
# --------------------------
if analysis_type == "Customer Segmentation" or analysis_type == "Dashboard Overview":
    st.header("🎯 Customer Segmentation Analysis")
    
    # Perform clustering
    features_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges']
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # Increased to 4 clusters for better segmentation
    df_encoded['Segment'] = kmeans.fit_predict(df_scaled[features_for_clustering])
    
    # Create meaningful segment labels based on characteristics
    segment_profiles = df_encoded.groupby('Segment')[features_for_clustering].mean()
    
    # Create better segment names
    segment_names = {}
    for i in range(4):
        tenure_avg = segment_profiles.loc[i, 'tenure']
        charges_avg = segment_profiles.loc[i, 'MonthlyCharges']
        if tenure_avg > df_encoded['tenure'].mean() and charges_avg > df_encoded['MonthlyCharges'].mean():
            segment_names[i] = 'Loyal High-Spenders'
        elif tenure_avg > df_encoded['tenure'].mean():
            segment_names[i] = 'Loyal Budget-Conscious'
        elif charges_avg > df_encoded['MonthlyCharges'].mean():
            segment_names[i] = 'New High-Spenders'
        else:
            segment_names[i] = 'New Budget-Conscious'
    
    df_encoded['Segment_Name'] = df_encoded['Segment'].map(segment_names)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Customer Segments")
        seg_counts = df_encoded['Segment_Name'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        wedges, texts, autotexts = ax.pie(seg_counts.values, labels=seg_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Customer Segment Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 Segment Characteristics")
        segment_details = df_encoded.groupby('Segment_Name').agg({
            'tenure': 'mean',
            'MonthlyCharges': 'mean',
            'TotalCharges': 'mean',
            'Segment': 'count'
        }).round(2)
        segment_details.columns = ['Avg Tenure', 'Avg Monthly Charges', 'Avg Total Charges', 'Count']
        st.dataframe(segment_details)
    
    st.subheader("💡 Recommended Marketing Strategies")
    strategies = {
        'Loyal High-Spenders': "🌟 VIP treatment, exclusive offers, premium support, loyalty rewards",
        'Loyal Budget-Conscious': "🎁 Volume discounts, referral bonuses, appreciation programs",
        'New High-Spenders': "🚀 Premium service upgrades, early access to new features",
        'New Budget-Conscious': "💰 Welcome discounts, free trials, gradual upselling"
    }
    
    for segment, strategy in strategies.items():
        if segment in seg_counts.index:
            st.write(f"**{segment}**: {strategy}")

# --------------------------
# Market Basket Analysis
# --------------------------
if analysis_type == "Market Basket Analysis" or analysis_type == "Dashboard Overview":
    st.header("🛒 Market Basket Analysis (Cross-Sell Opportunities)")
    
    # Select boolean/binary service columns
    service_cols = [col for col in df_original.columns if df_original[col].nunique() == 2 and col != 'Churn']
    
    if len(service_cols) > 0:
        st.subheader("📊 Service Adoption Analysis")
        
        # Convert to boolean for market basket analysis
        df_basket = df_original[service_cols].copy()
        
        # Handle different encoding formats (Yes/No, 1/0)
        for col in service_cols:
            unique_vals = df_basket[col].unique()
            if 'Yes' in unique_vals:
                df_basket[col] = (df_basket[col] == 'Yes')
            elif 'No' in unique_vals:
                df_basket[col] = (df_basket[col] != 'No')
            else:
                df_basket[col] = df_basket[col].astype(bool)
        
        # Generate frequent itemsets
        try:
            frequent_itemsets = apriori(df_basket, min_support=0.1, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
                
                if len(rules) > 0:
                    # Convert frozensets to strings
                    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    # Display top rules
                    st.subheader("🔗 Top Cross-Sell Opportunities")
                    top_rules = rules.nlargest(10, 'lift')[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
                    top_rules.columns = ['If Customer Has', 'Recommend', 'Support', 'Confidence', 'Lift']
                    st.dataframe(top_rules.round(3))
                    
                    # Service adoption heatmap
                    st.subheader("🔥 Service Adoption Heatmap")
                    corr_matrix = df_basket.corr()
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Service Correlation Matrix')
                    st.pyplot(fig)
                    
                    # Marketing insights
                    st.subheader("💡 Key Marketing Insights")
                    if len(rules) > 0:
                        best_rule = rules.loc[rules['lift'].idxmax()]
                        st.success(f"🎯 **Best Cross-sell Opportunity**: If customer has {list(best_rule['antecedents'])[0]}, recommend {list(best_rule['consequents'])[0]} (Lift: {best_rule['lift']:.2f})")
                        
                        st.write("**Recommended Actions:**")
                        for _, rule in rules.head(3).iterrows():
                            antecedent = list(rule['antecedents'])[0]
                            consequent = list(rule['consequents'])[0]
                            st.write(f"• Target customers with **{antecedent}** for **{consequent}** promotions")
                else:
                    st.warning("No significant association rules found. Try lowering the minimum threshold.")
            else:
                st.warning("No frequent itemsets found. The data might be too sparse.")
        except Exception as e:
            st.error(f"Error in market basket analysis: {str(e)}")
    else:
        st.warning("No suitable service columns found for market basket analysis.")

# --------------------------
# A/B Testing Simulation
# --------------------------
if analysis_type == "A/B Testing" or analysis_type == "Dashboard Overview":
    st.header("🧪 A/B Testing Simulation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Test Configuration")
        
        # Test parameters
        group_size = st.number_input("Sample Size per Group", min_value=100, max_value=10000, value=1000, step=100)
        
        strategy_A = st.text_input("Strategy A Name", value="Discount Offer")
        conversion_rate_A = st.slider("Strategy A Conversion Rate", 0.01, 0.50, 0.15, 0.01)
        
        strategy_B = st.text_input("Strategy B Name", value="Premium Support")
        conversion_rate_B = st.slider("Strategy B Conversion Rate", 0.01, 0.50, 0.20, 0.01)
        
        # Run simulation
        np.random.seed(42)  # For reproducible results
        conversions_A = np.random.binomial(group_size, conversion_rate_A)
        conversions_B = np.random.binomial(group_size, conversion_rate_B)
        
        # Statistical significance test (simplified)
        difference = abs(conversions_B - conversions_A)
        relative_improvement = (conversions_B - conversions_A) / conversions_A * 100 if conversions_A > 0 else 0
    
    with col2:
        st.subheader("📈 Results")
        
        # Display metrics
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric(label=strategy_A, value=f"{conversions_A}", delta=f"{conversions_A/group_size*100:.1f}% conversion")
        
        with col2_2:
            st.metric(label=strategy_B, value=f"{conversions_B}", delta=f"{conversions_B/group_size*100:.1f}% conversion")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        strategies = [strategy_A, strategy_B]
        conversions = [conversions_A, conversions_B]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(strategies, conversions, color=colors, alpha=0.8)
        ax.set_ylabel("Number of Conversions")
        ax.set_title("A/B Test Results")
        
        # Add value labels on bars
        for bar, conv in zip(bars, conversions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{conv}\n({conv/group_size*100:.1f}%)', 
                   ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Statistical conclusion
        if conversions_B > conversions_A:
            st.success(f"✅ **{strategy_B} Wins!** (+{relative_improvement:.1f}% improvement)")
        elif conversions_A > conversions_B:
            st.success(f"✅ **{strategy_A} Wins!** (+{abs(relative_improvement):.1f}% better)")
        else:
            st.info("🤝 **It's a Tie!** Both strategies performed equally")
        
        # Confidence interval (simplified)
        st.write("**📊 Statistical Summary:**")
        st.write(f"• Sample size: {group_size:,} per group")
        st.write(f"• Absolute difference: {difference} conversions")
        st.write(f"• Relative improvement: {relative_improvement:.1f}%")

# --------------------------
# Dashboard Overview
# --------------------------
if analysis_type == "Dashboard Overview":
    st.header("📋 Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df_encoded)
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col2:
        avg_clv = df_encoded['CLV'].mean()
        st.metric("💰 Avg CLV", f"${avg_clv:.2f}")
    
    with col3:
        avg_tenure = df_encoded['tenure'].mean()
        st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_monthly = df_encoded['MonthlyCharges'].mean()
        st.metric("💳 Avg Monthly", f"${avg_monthly:.2f}")
    
    st.markdown("---")
    st.write("**💡 Key Insights:**")
    st.write("• Customer segmentation reveals distinct spending patterns for targeted marketing")
    st.write("• Market basket analysis identifies cross-selling opportunities")
    st.write("• CLV prediction helps prioritize high-value prospects")
    st.write("• A/B testing framework enables data-driven campaign decisions")