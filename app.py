import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import inspect

st.set_page_config(page_title="EA Attrition Dashboard", layout="wide")

# -----------------------------
# Load data
# -----------------------------
def load_data():
    uploaded = st.file_uploader("Upload your EA.csv dataset", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv("/mnt/data/EA.csv")
    return df

# -----------------------------
# Preprocessor
# -----------------------------
def get_preprocessor(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(exclude=['number']).columns if df[c].nunique() > 1]

    if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

    num_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('onehot', onehot)
    ])

    preprocessor = ColumnTransformer([
        ('num', num_trans, num_cols),
        ('cat', cat_trans, cat_cols)
    ])

    return preprocessor, num_cols, cat_cols

# -----------------------------
# Model training
# -----------------------------
def train_models(df):
    df.columns = df.columns.str.strip()
    if "Attrition" not in df.columns:
        raise ValueError("The dataset must contain an 'Attrition' column.")

    y = df["Attrition"]
    if y.dtype == "object":
        y = y.str.strip().str.lower().map({"yes": 1, "no": 0})
    y = y.fillna(0).astype(int)
    X = df.drop(columns=["Attrition"])

    if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }

    results = []
    confs = {}
    rocs = {}

    for name, model in models.items():
        preproc, _, _ = get_preprocessor(X)
        pipe = Pipeline([("preproc", preproc), ("clf", model)])
        cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=1, error_score='raise')
        y_pred = cross_val_predict(pipe, X, y, cv=cv)

        cm = confusion_matrix(y, y_pred)
        confs[name] = cm

        # Compute ROC AUC
        pipe.fit(X, y)
        y_proba = pipe.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        rocs[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}

        results.append({
            "Model": name,
            "Train Accuracy": cv_res["train_accuracy"].mean(),
            "Test Accuracy": cv_res["test_accuracy"].mean(),
            "Precision": cv_res["test_precision"].mean(),
            "Recall": cv_res["test_recall"].mean(),
            "F1": cv_res["test_f1"].mean(),
            "AUC": cv_res["test_roc_auc"].mean()
        })

    res_df = pd.DataFrame(results)
    return res_df, confs, rocs, preproc

# -----------------------------
# Utilities
# -----------------------------
def plot_confusion_matrix_bw(cm, labels=['Stayed (0)','Left (1)']):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(cm, cmap='Greys', interpolation='nearest')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=12)
    plt.tight_layout()
    return fig

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Employee Attrition — Streamlit Dashboard")
st.markdown("Interactive dashboard for EDA and modelling. Default dataset: `/mnt/data/EA.csv`")

# Sidebar filters
st.sidebar.header("Filters")
df = load_data()
all_roles = sorted(df['JobRole'].dropna().unique().tolist())
selected_roles = st.sidebar.multiselect("Filter by Job Role (multi-select)", options=all_roles, default=all_roles)
satisfaction_cols = [c for c in df.columns if 'Satisfaction' in c or 'Satis' in c]
s_col = st.sidebar.selectbox("Satisfaction column for slider", options=satisfaction_cols)
min_val = int(df[s_col].min()); max_val = int(df[s_col].max())
s_range = st.sidebar.slider(f"{s_col} range", min_value=min_val, max_value=max_val, value=(min_val, max_val))

df_filt = df[df['JobRole'].isin(selected_roles)]
df_filt = df_filt[(df_filt[s_col]>=s_range[0]) & (df_filt[s_col]<=s_range[1])]

tab1, tab2, tab3 = st.tabs(["Dashboard (EDA)", "Models (Train & Evaluate)", "Predict / Upload"])

# -----------------------------
# Tab 1: EDA
# -----------------------------
with tab1:
    st.header("Exploratory Insights — Top Charts")
    grp = df_filt.groupby('JobRole').agg(total=('Attrition','count'), left=('Attrition', lambda x: (x=='Yes').sum()))
    grp = grp.assign(attrition_rate = grp['left'] / grp['total'] * 100).reset_index().sort_values('attrition_rate', ascending=False)
    fig1 = px.bar(grp, x='JobRole', y='attrition_rate', hover_data=['total','left'], title='Attrition Rate by Job Role (%)')
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.violin(df_filt, y='MonthlyIncome', x='Attrition', box=True, points='all', title='Monthly Income by Attrition')
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.scatter(df_filt, x='Age', y='YearsAtCompany', color='Attrition', size='MonthlyIncome', hover_data=['JobRole','JobLevel'], title='Age vs YearsAtCompany (size=MonthlyIncome)')
    st.plotly_chart(fig3, use_container_width=True)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr = df_filt[num_cols].corr()
    fig4 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Greys', reversescale=True))
    fig4.update_layout(title='Correlation matrix (numeric features) — greyscale')
    st.plotly_chart(fig4, use_container_width=True)
    sat_grp = df_filt.groupby([s_col, 'Attrition']).size().reset_index(name='count')
    fig5 = px.bar(sat_grp, x=s_col, y='count', color='Attrition', barmode='group', title=f'{s_col} vs Attrition counts')
    st.plotly_chart(fig5, use_container_width=True)
    st.subheader("Quick Dataset Summary (filtered)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df_filt))
    c2.metric("Attrition rate (%)", round((df_filt['Attrition']=='Yes').mean()*100,2))
    c3.metric("Mean Monthly Income", int(df_filt['MonthlyIncome'].mean()))
    c4.metric("Mean YearsAtCompany", round(df_filt['YearsAtCompany'].mean(),2))

# -----------------------------
# Tab 2: Model Training
# -----------------------------
with tab2:
    st.header("Model Training & Evaluation")
    if 'model_results' not in st.session_state:
        st.session_state['model_results'] = None
    run_btn = st.button("Run Models (5-fold CV)")
    if run_btn:
        with st.spinner("Training models (this may take a minute)..."):
            res_df, confs, rocs, preproc = train_models(df)
            st.session_state['model_results'] = (res_df, confs, rocs, preproc)
            st.success("Training complete. Results stored in session.")
    if st.session_state['model_results'] is not None:
        res_df, confs, rocs, preproc = st.session_state['model_results']
        st.subheader("Performance summary (averaged across folds)")
        st.dataframe(res_df.round(4))
        st.subheader("Confusion Matrices (grayscale)")
        if confs is None:
            st.warning("No confusion matrices available.")
        elif isinstance(confs, dict):
            for name, cm in confs.items():
                st.write(f"**{name} Confusion Matrix**")
                st.dataframe(pd.DataFrame(cm).round(4))
        elif isinstance(confs, (list, tuple, np.ndarray)):
            if isinstance(confs, np.ndarray) and confs.ndim == 2:
                confs = [confs]
            for i, cm in enumerate(confs):
                st.write(f"**Confusion Matrix {i+1}**")
                st.dataframe(pd.DataFrame(cm).round(4))
        else:
            st.error(f"Unsupported type for confusion matrices: {type(confs)}")

        st.subheader("ROC Curves")
        fig = go.Figure()
        for name, info in rocs.items():
            fig.add_trace(go.Scatter(x=info['fpr'], y=info['tpr'], mode='lines', name=f"{name} (AUC={info['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
        fig.update_layout(title='ROC curves (combined)', xaxis_title='FPR', yaxis_title='TPR', legend=dict(x=0.7,y=0.1))
        st.plotly_chart(fig, use_container_width=True)
        csv = to_csv_bytes(res_df)
        st.download_button("Download performance summary CSV", data=csv, file_name="EA_model_summary.csv", mime="text/csv")

# -----------------------------
# Tab 3: Predict on new data
# -----------------------------
with tab3:
    st.header("Upload new data & Predict Attrition")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        newdf = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(newdf.head())
        if st.button("Predict on uploaded data"):
            if st.session_state.get('model_results') is None:
                st.info("Training models on full dataset first...")
                res_df, confs, rocs, preproc = train_models(df)
            else:
                res_df, confs, rocs, preproc = st.session_state['model_results']
            model = GradientBoostingClassifier(n_estimators=200, random_state=42)
            pipe = Pipeline([('preproc', preproc), ('clf', model)])
            if 'Attrition' in newdf.columns:
                newX = newdf.drop(columns=['Attrition'])
            else:
                newX = newdf.copy()
            y_full = df['Attrition'].astype(str)
            label_map = {v:i for i,v in enumerate(y_full.unique().tolist())}
            pipe.fit(df.drop(columns=['Attrition']), y_full.map(label_map).values)
            preds = pipe.predict(newX)
            try:
                pred_labels = ['Yes' if p==1 else 'No' for p in preds]
            except:
                pred_labels = preds.astype(str)
            newdf['Predicted_Attrition'] = pred_labels
            st.write(newdf.head())
            csvbytes = to_csv_bytes(newdf)
            st.download_button("Download predicted CSV", data=csvbytes, file_name="EA_predicted.csv", mime="text/csv")
