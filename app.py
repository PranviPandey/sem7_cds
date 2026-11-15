import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG / THEME ----------------
st.set_page_config(page_title="Student Performance Intelligence", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp{background:#080D1A;color:#E2E8F0}
.block-container{padding:2rem 2.5rem 3rem;max-width:1650px}
p,li,label{color:#CBD5E1}
section[data-testid="stSidebar"]{background:#111827;border-right:1px solid #263D5D}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#38BDF8}
.dashboard-title{font-size:34px;font-weight:800;color:#38BDF8;margin-bottom:7px;text-shadow:0 0 18px #38bdf833}
.dashboard-subtitle{color:#94A3B8;font-size:15px;margin-bottom:22px}
.section-heading{font-size:22px;font-weight:750;color:#7DD3FC;margin:12px 0 15px}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#17243A,#101A2B);padding:20px;border-radius:14px;border:1px solid #2B4261;min-height:118px}
div[data-testid="stMetric"]:hover{border-color:#38BDF8;box-shadow:0 0 16px #38bdf822}
div[data-testid="stMetricLabel"]{color:#94A3B8!important}
div[data-testid="stMetricValue"]{color:#38BDF8!important;font-weight:800}
div[data-testid="stVerticalBlockBorderWrapper"],div[data-testid="stPlotlyChart"]{background:#111C2E;border:1px solid #263D5D;border-radius:13px;padding:8px}
div[data-testid="stDataFrame"]{border:1px solid #263D5D;border-radius:10px}
.insight-card{background:linear-gradient(120deg,#17243A,#101B2D);border:1px solid #263D5D;border-left:4px solid #38BDF8;border-radius:11px;padding:15px 18px;margin-bottom:12px;color:#CBD5E1;line-height:1.6}
.insight-card b{color:#7DD3FC}
hr{border-color:#263D5D!important;margin:20px 0}
.stButton>button{background:#0C4A6E;color:#E0F2FE;border:1px solid #0EA5E9;border-radius:8px}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

BLUE, CYAN, PURPLE, PINK, PANEL, GRID = "#38BDF8", "#22D3EE", "#818CF8", "#F472B6", "#111C2E", "#263D5D"
TARGET = "Exam_Score"

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    path = Path(__file__).resolve().parent/ "StudentPerformanceFactors.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    for c in raw.select_dtypes(include="object"):
        raw[c] = raw[c].str.strip()
    missing, duplicates = int(raw.isna().sum().sum()), int(raw.duplicated().sum())
    data = raw.drop_duplicates().copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    for c in data.select_dtypes(include=np.number):
        med = data[c].median()
        if pd.notna(med): data[c] = data[c].fillna(med)
    for c in data.select_dtypes(exclude=np.number):
        mode = data[c].mode()
        if not mode.empty: data[c] = data[c].fillna(mode.iloc[0])
    return data, raw, missing, duplicates

try:
    df, raw_df, raw_missing, raw_duplicates = load_data()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()
if TARGET not in df:
    st.error(f"Missing required column: {TARGET}")
    st.stop()

numeric = df.select_dtypes(include=np.number).columns.tolist()
predictors = [c for c in numeric if c != TARGET]
categorical = df.select_dtypes(exclude=np.number).columns.tolist()

# ---------------- HELPERS ----------------
def heading(t):
    st.markdown(f'<div class="section-heading">{t}</div>', unsafe_allow_html=True)

def insight(title, text):
    st.markdown(f'<div class="insight-card"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)

def ptxt(p):
    return "N/A" if pd.isna(p) else ("<0.001" if p < .001 else f"{p:.4f}")

def strength(r):
    a = abs(r)
    return "negligible" if a < .1 else "weak" if a < .3 else "moderate" if a < .5 else "strong" if a < .7 else "very strong"

def chart(fig, height=400):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color="#CBD5E1"), title_font=dict(color=BLUE, size=18),
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), height=height,
        margin=dict(l=40,r=30,t=65,b=40), legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def mean_ci(x, confidence=.95):
    x = pd.Series(x).dropna()
    if len(x) < 2: return np.nan, np.nan
    return stats.t.interval(confidence, len(x)-1, loc=x.mean(), scale=stats.sem(x))

@st.cache_data
def correlations(data):
    rows = []
    for col in [c for c in data.select_dtypes(include=np.number) if c != TARGET]:
        pair = data[[col,TARGET]].dropna()
        if len(pair) >= 3 and pair[col].nunique() > 1:
            r,p = stats.pearsonr(pair[col],pair[TARGET])
            rho,sp = stats.spearmanr(pair[col],pair[TARGET])
            rows.append({"Variable":col,"Pearson r":r,"Pearson p":p,
                         "Spearman rho":rho,"Spearman p":sp,"N":len(pair)})
    out = pd.DataFrame(rows)
    if len(out):
        reject,adj,_,_ = multipletests(out["Pearson p"],method="holm")
        out["Holm adjusted p"],out["Holm significant"] = adj,reject
    return out

corr = correlations(df)

def make_insights(data, ctable):
    items = []
    if len(ctable):
        row = ctable.iloc[ctable["Pearson r"].abs().argmax()]
        items.append(("Strongest observed association",
            f"{row['Variable']} has the largest absolute Pearson correlation with exam score "
            f"(r={row['Pearson r']:.3f}, p={ptxt(row['Pearson p'])}, N={int(row['N']):,}). "
            f"This is a {strength(row['Pearson r'])} association, not evidence of causation."))
        sig = ctable[ctable["Holm significant"]]
        items.append(("Multiple-testing result",
            f"{len(sig)} of {len(ctable)} numerical correlations remain significant after "
            f"Holm correction: {', '.join(sig['Variable']) if len(sig) else 'none'}."))
        sleep = ctable[ctable.Variable.str.lower()=="sleep_hours"]
        if len(sleep) and sleep.iloc[0]["Pearson p"] >= .05:
            row=sleep.iloc[0]
            items.append(("Non-significant finding",
                f"Sleep hours: r={row['Pearson r']:.3f}, p={ptxt(row['Pearson p'])}. "
                "This analysis does not establish a significant linear association."))
    items.append(("Data quality",
        f"The raw dataset contained {raw_missing:,} missing cells and {raw_duplicates:,} duplicate rows. "
        "This app applies median/mode imputation after dropping rows with missing exam scores."))
    return items

def make_recommendations(data, ctable):
    rows=[]
    for _,r in ctable.iterrows():
        if r["Pearson p"] >= .05: continue
        a=abs(r["Pearson r"])
        priority="High" if a>=.5 else "Medium" if a>=.3 else "Low"
        practical=("Comparatively strong association; worth investigating." if a>=.5 else
                   "Moderate association; investigate alongside other factors." if a>=.3 else
                   "Weak association; statistical significance may have limited practical importance.")
        action=(f"Review {r['Variable']} patterns across performance groups and consider whether "
                "support or monitoring is appropriate. Evaluate outcomes before claiming an intervention works.")
        rows.append({"Factor":r["Variable"],"Evidence":f"r={r['Pearson r']:.3f}; raw p={ptxt(r['Pearson p'])}; Holm p={ptxt(r['Holm adjusted p'])}",
                     "Priority":priority,"Interpretation":practical,"Suggested action":action})
    for col in ["Motivation_Level","Parental_Involvement","Access_to_Resources","Teacher_Quality","Internet_Access"]:
        if col not in data: continue
        g=data.groupby(col,observed=True)[TARGET].agg(["count","mean"])
        if len(g)<2: continue
        hi,lo=g["mean"].idxmax(),g["mean"].idxmin()
        rows.append({"Factor":col,
            "Evidence":f"Highest mean: {hi} ({g.loc[hi,'mean']:.2f}); lowest: {lo} ({g.loc[lo,'mean']:.2f}); gap={g['mean'].max()-g['mean'].min():.2f}.",
            "Priority":"Investigate","Interpretation":"Descriptive group difference; it does not show that group membership causes performance.",
            "Suggested action":f"Review group sizes and context for {col}; investigate whether support or resource needs differ."})
    return pd.DataFrame(rows)

insights = make_insights(df,corr)
recs = make_recommendations(df,corr)

# ---------------- NAVIGATION ----------------
pages=["Overview","Exploratory Analysis","Statistical Analysis","Regression Analysis",
       "Student Group Insights","Recommendations & Reports"]
with st.sidebar:
    st.markdown('<div style="font-size:25px;font-weight:800;color:#38BDF8;">🎓 Student Analytics</div><div style="color:#94A3B8;font-size:12px;">Performance Intelligence System</div>',unsafe_allow_html=True)
    st.divider()
    page=st.radio("NAVIGATION",pages,label_visibility="collapsed")
    st.divider()
    st.metric("Cleaned records",f"{len(df):,}")
    st.metric("Attributes",df.shape[1])
    st.caption("Student Performance Factors")
    st.divider()
    st.caption("Computing for Data Science")

st.markdown(f'<div class="dashboard-title">{page}</div><div class="dashboard-subtitle">Student Academic Performance Intelligence & Decision Support</div>',unsafe_allow_html=True)

# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================
if page=="Overview":
    heading("Performance Overview")
    cols=st.columns(4)
    for col,(label,value) in zip(cols,[
        ("Total records",f"{len(df):,}"),("Average exam score",f"{df[TARGET].mean():.2f}"),
        ("Average attendance",f"{df['Attendance'].mean():.2f}" if "Attendance" in df else "N/A"),
        ("Average study hours",f"{df['Hours_Studied'].mean():.2f}" if "Hours_Studied" in df else "N/A")]):
        col.metric(label,value)
    st.divider()
    heading("Dataset Profile")
    a,b=st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("### Data quality")
            st.dataframe(pd.DataFrame({"Metric":["Raw records","Cleaned records","Attributes","Numeric columns","Categorical columns","Raw missing cells","Cleaned missing cells","Raw duplicates"],
                "Value":[len(raw_df),len(df),df.shape[1],len(numeric),len(categorical),raw_missing,int(df.isna().sum().sum()),raw_duplicates]}),
                hide_index=True,use_container_width=True)
    with b:
        with st.container(border=True):
            st.markdown("### Exam score statistics")
            st.dataframe(df[TARGET].describe().round(3).rename_axis("Statistic").reset_index(),
                         hide_index=True,use_container_width=True)
    st.divider()
    heading("Examination Score Distribution")
    with st.container(border=True):
        fig=px.histogram(df,x=TARGET,nbins=30,marginal="box",color_discrete_sequence=[BLUE],title="Exam Score Distribution")
        fig.add_vline(x=df[TARGET].mean(),line_dash="dash",line_color=PINK,annotation_text="Mean")
        st.plotly_chart(chart(fig,420),use_container_width=True)
    heading("Academic Factors")
    a,b=st.columns(2)
    for col,x,color,title in [(a,"Hours_Studied",BLUE,"Study hours vs score"),(b,"Attendance",CYAN,"Attendance vs score")]:
        if x in df:
            with col:
                with st.container(border=True):
                    fig=px.scatter(df,x=x,y=TARGET,opacity=.55,color_discrete_sequence=[color],title=title)
                    st.plotly_chart(chart(fig,370),use_container_width=True)
    heading("Automatically Detected Insights")
    for title,text in insights: insight(title,text)
    st.caption("The dataset is synthetic. Results describe this dataset and should not be generalized to real student populations.")

# ============================================================
# PAGE 2 - EXPLORATORY ANALYSIS
# ============================================================
elif page=="Exploratory Analysis":
    heading("Explore Numerical Variables")
    var=st.selectbox("Numerical variable",numeric,index=numeric.index(TARGET))
    s=df[var]; q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1
    out=s[(s<q1-1.5*iqr)|(s>q3+1.5*iqr)]
    c=st.columns(4)
    for col,(label,val) in zip(c,[("Mean",f"{s.mean():.2f}"),("Median",f"{s.median():.2f}"),("Std. deviation",f"{s.std():.2f}"),("IQR outliers",f"{len(out):,}")]): col.metric(label,val)
    a,b=st.columns(2)
    with a:
        with st.container(border=True):
            st.plotly_chart(chart(px.histogram(df,x=var,nbins=30,marginal="rug",color_discrete_sequence=[BLUE],title=f"Distribution: {var}")),use_container_width=True)
    with b:
        with st.container(border=True):
            st.plotly_chart(chart(px.box(df,y=var,color_discrete_sequence=[PURPLE],title=f"Boxplot: {var}")),use_container_width=True)
    heading("Variable Relationships")
    a,b=st.columns(2)
    x=a.selectbox("X variable",numeric,index=0,key="eda_x")
    y=b.selectbox("Y variable",numeric,index=numeric.index(TARGET),key="eda_y")
    if x!=y:
        pair=df[[x,y]].dropna(); r,p=stats.pearsonr(pair[x],pair[y]); rho,sp=stats.spearmanr(pair[x],pair[y])
        c=st.columns(3); c[0].metric("Pearson r",f"{r:.4f}"); c[1].metric("Spearman rho",f"{rho:.4f}"); c[2].metric("Pearson p",ptxt(p))
        fig=px.scatter(pair,x=x,y=y,trendline="ols",opacity=.5,color_discrete_sequence=[BLUE],title=f"{x} vs {y}")
        st.plotly_chart(chart(fig,430),use_container_width=True)
    else: st.info("Choose two different variables.")
    heading("Correlation Heatmap")
    method=st.selectbox("Method",["pearson","spearman"])
    fig=px.imshow(df[numeric].corr(method=method),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title=f"{method.title()} correlation matrix")
    st.plotly_chart(chart(fig,620),use_container_width=True)
    heading("Categorical Comparisons")
    if categorical:
        cat=st.selectbox("Categorical variable",categorical)
        tab=df.groupby(cat,observed=True)[TARGET].agg(["count","mean","median","std"]).reset_index()
        st.dataframe(tab.round(3),hide_index=True,use_container_width=True)
        st.plotly_chart(chart(px.box(df,x=cat,y=TARGET,color=cat,title=f"Exam score by {cat}"),420),use_container_width=True)

# ============================================================
# PAGE 3 - STATISTICAL ANALYSIS
# ============================================================
elif page=="Statistical Analysis":
    heading("Inferential Statistical Analysis")
    tabs=st.tabs(["Correlation","Welch t-test","ANOVA","Chi-square","Confidence interval"])
    with tabs[0]:
        st.dataframe(corr.round(5),hide_index=True,use_container_width=True)
        fig=px.bar(corr.sort_values("Pearson r"),x="Pearson r",y="Variable",orientation="h",color="Pearson r",color_continuous_scale="RdBu",title="Pearson correlation with Exam Score")
        st.plotly_chart(chart(fig,430),use_container_width=True)
    with tabs[1]:
        binary=[c for c in categorical if df[c].nunique()==2]
        if binary:
            cat=st.selectbox("Binary grouping variable",binary)
            groups=list(df[cat].dropna().unique()); g1=df.loc[df[cat]==groups[0],TARGET]; g2=df.loc[df[cat]==groups[1],TARGET]
            t,p=stats.ttest_ind(g1,g2,equal_var=False); pooled=np.sqrt((g1.var(ddof=1)+g2.var(ddof=1))/2); d=(g1.mean()-g2.mean())/pooled if pooled else np.nan
            c=st.columns(3); c[0].metric("Welch t",f"{t:.4f}"); c[1].metric("P-value",ptxt(p)); c[2].metric("Cohen's d",f"{d:.4f}")
            st.dataframe(pd.DataFrame({"Group":groups,"N":[len(g1),len(g2)],"Mean":[g1.mean(),g2.mean()],"SD":[g1.std(),g2.std()]}).round(3),hide_index=True,use_container_width=True)
            st.plotly_chart(chart(px.box(df,x=cat,y=TARGET,color=cat,title=f"Exam score by {cat}")),use_container_width=True)
        else: st.info("No binary categorical variable found.")
    with tabs[2]:
        multi=[c for c in categorical if 2<df[c].nunique()<=20]
        if multi:
            cat=st.selectbox("ANOVA grouping variable",multi)
            groups=[g[TARGET].dropna().values for _,g in df.groupby(cat,observed=True) if len(g)>1]
            f,p=stats.f_oneway(*groups); lev,lp=stats.levene(*groups)
            grand=df[TARGET].mean(); ssb=sum(len(g)*(g.mean()-grand)**2 for g in groups); sst=((df[TARGET]-grand)**2).sum(); eta=ssb/sst if sst else np.nan
            c=st.columns(3); c[0].metric("F statistic",f"{f:.4f}"); c[1].metric("P-value",ptxt(p)); c[2].metric("Eta squared",f"{eta:.4f}")
            st.write(f"Levene statistic={lev:.4f}; p={ptxt(lp)}")
            summary=df.groupby(cat,observed=True)[TARGET].agg(["count","mean","std"]).reset_index()
            st.dataframe(summary.round(3),hide_index=True,use_container_width=True)
            st.plotly_chart(chart(px.box(df,x=cat,y=TARGET,color=cat,title=f"Exam score by {cat}")),use_container_width=True)
            if p<.05:
                tukey=pairwise_tukeyhsd(df[TARGET],df[cat],alpha=.05)
                st.markdown("#### Tukey HSD")
                st.dataframe(pd.DataFrame(tukey.summary().data[1:],columns=tukey.summary().data[0]),hide_index=True,use_container_width=True)
        else: st.info("No suitable categorical variable.")
    with tabs[3]:
        if len(categorical)>=2:
            a,b=st.columns(2); v1=a.selectbox("First variable",categorical); other=[x for x in categorical if x!=v1]; v2=b.selectbox("Second variable",other)
            table=pd.crosstab(df[v1],df[v2]); chi,p,dof,expected=chi2_contingency(table); n=table.values.sum(); denom=min(table.shape)-1; cv=np.sqrt(chi/(n*denom)) if denom>0 else np.nan
            c=st.columns(3); c[0].metric("Chi-square",f"{chi:.4f}"); c[1].metric("P-value",ptxt(p)); c[2].metric("Cramer's V",f"{cv:.4f}")
            st.dataframe(table,use_container_width=True); st.write(f"Degrees of freedom: {dof}; minimum expected count: {expected.min():.2f}")
    with tabs[4]:
        var=st.selectbox("Variable",numeric,key="ci_var"); conf=st.select_slider("Confidence level",[.90,.95,.99],value=.95); vals=df[var].dropna(); low,high=mean_ci(vals,conf)
        c=st.columns(3); c[0].metric("Mean",f"{vals.mean():.3f}"); c[1].metric("Lower",f"{low:.3f}"); c[2].metric("Upper",f"{high:.3f}")
        fig=go.Figure(go.Scatter(x=[low,high],y=[0,0],mode="lines+markers",line=dict(color=BLUE,width=5),marker=dict(size=12)))
        fig.add_vline(x=vals.mean(),line_dash="dash",line_color=PINK); fig.update_layout(title=f"{conf:.0%} confidence interval: {var}",xaxis_title=var,yaxis=dict(visible=False))
        st.plotly_chart(chart(fig,270),use_container_width=True)

# ============================================================
# PAGE 4 - REGRESSION
# ============================================================
elif page=="Regression Analysis":
    heading("Multiple Linear Regression")
    required=["Exam_Score","Hours_Studied","Attendance","Sleep_Hours","Previous_Scores","Tutoring_Sessions","Physical_Activity","Motivation_Level","Parental_Involvement","Access_to_Resources","Teacher_Quality","Internet_Access"]
    absent=[c for c in required if c not in df]
    if absent: st.error(f"Regression columns missing: {absent}")
    else:
        formula=("Exam_Score ~ Hours_Studied + Attendance + Sleep_Hours + Previous_Scores + Tutoring_Sessions + Physical_Activity + C(Motivation_Level) + C(Parental_Involvement) + C(Access_to_Resources) + C(Teacher_Quality) + C(Internet_Access)")
        md=df[required].dropna(); model=ols(formula,data=md).fit(); robust=ols(formula,data=md).fit(cov_type="HC3")
        resid=model.resid; fitted=model.fittedvalues
        c=st.columns(4)
        for col,(lab,val) in zip(c,[("R-squared",f"{model.rsquared:.4f}"),("Adjusted R²",f"{model.rsquared_adj:.4f}"),("RMSE",f"{np.sqrt(np.mean(resid**2)):.3f}"),("MAE",f"{np.mean(np.abs(resid)):.3f}")]): col.metric(lab,val)
        heading("Model Summary")
        st.dataframe(pd.DataFrame({"Metric":["N","R²","Adjusted R²","F statistic","F-test p","AIC","BIC"],"Value":[model.nobs,model.rsquared,model.rsquared_adj,model.fvalue,model.f_pvalue,model.aic,model.bic]}).round(5),hide_index=True,use_container_width=True)
        heading("Coefficients with HC3 Robust Standard Errors")
        ci=robust.conf_int()
        coef=pd.DataFrame({"Variable":robust.params.index,"Coefficient":robust.params.values,"Robust SE":robust.bse.values,"P-value":robust.pvalues.values,"CI lower":ci.iloc[:,0].values,"CI upper":ci.iloc[:,1].values})
        st.dataframe(coef.round(5),hide_index=True,use_container_width=True)
        st.plotly_chart(chart(px.bar(coef[coef.Variable!="Intercept"].sort_values("Coefficient"),x="Coefficient",y="Variable",orientation="h",color="Coefficient",color_continuous_scale="RdBu",title="Regression Coefficients"),520),use_container_width=True)
        heading("Residual Diagnostics")
        a,b=st.columns(2)
        with a:
            fig=px.scatter(x=fitted,y=resid,labels={"x":"Fitted","y":"Residual"},title="Residuals vs fitted"); fig.add_hline(y=0,line_dash="dash",line_color=PINK); st.plotly_chart(chart(fig,360),use_container_width=True)
        with b:
            prob=stats.probplot(resid,dist="norm"); theoretical,ordered=prob[0]; slope,intercept,_=prob[1]
            fig=go.Figure(); fig.add_trace(go.Scatter(x=theoretical,y=ordered,mode="markers",marker=dict(color=BLUE,size=5))); fig.add_trace(go.Scatter(x=theoretical,y=slope*theoretical+intercept,mode="lines",line=dict(color=PINK,dash="dash")))
            fig.update_layout(title="Normal Q-Q Plot",xaxis_title="Theoretical quantiles",yaxis_title="Ordered residuals"); st.plotly_chart(chart(fig,360),use_container_width=True)
        heading("Multicollinearity: VIF")
        exog=pd.DataFrame(model.model.exog,columns=model.model.exog_names); vr=[]
        for i,col in enumerate(exog.columns):
            if col!="Intercept":
                try: value=variance_inflation_factor(exog.values,i)
                except Exception: value=np.nan
                vr.append({"Variable":col,"VIF":value})
        st.dataframe(pd.DataFrame(vr).round(3),hide_index=True,use_container_width=True)
        heading("Breusch-Pagan Test")
        bp=het_breuschpagan(resid,model.model.exog); c=st.columns(2); c[0].metric("LM statistic",f"{bp[0]:.4f}"); c[1].metric("P-value",ptxt(bp[1]))
        heading("Cook's Distance")
        cooks=model.get_influence().cooks_distance[0]; threshold=4/len(md); count=int((cooks>threshold).sum())
        c=st.columns(2); c[0].metric("4/n threshold",f"{threshold:.6f}"); c[1].metric("Observations above threshold",count)
        fig=px.scatter(x=np.arange(len(cooks)),y=cooks,labels={"x":"Observation index","y":"Cook's distance"},title="Influential observations"); fig.add_hline(y=threshold,line_dash="dash",line_color=PINK)
        st.plotly_chart(chart(fig,350),use_container_width=True)
        st.caption("HC3 robust standard errors affect coefficient inference; they do not fix model misspecification or influential observations. Cook's distance is a screening tool, not an automatic deletion rule.")

# ============================================================
# PAGE 5 - STUDENT GROUP INSIGHTS
# ============================================================
elif page=="Student Group Insights":
    heading("Performance Segmentation")
    st.write("Compare descriptive performance patterns across groups. These are analytical segments, not official student risk classifications.")
    choices=[c for c in ["Attendance","Hours_Studied","Previous_Scores","Sleep_Hours","Tutoring_Sessions"] if c in df]
    var=st.selectbox("Segment by",choices); cutoff=float(df[var].median())
    seg=df.copy(); seg["Segment"]=np.where(seg[var]>=cutoff,f"Higher {var}",f"Lower {var}")
    c=st.columns(3); c[0].metric("Median split",f"{cutoff:.2f}"); c[1].metric("Higher group",f"{(seg['Segment']==f'Higher {var}').sum():,}"); c[2].metric("Lower group",f"{(seg['Segment']==f'Lower {var}').sum():,}")
    summary=seg.groupby("Segment")[TARGET].agg(["count","mean","median","std"]).reset_index()
    summary.columns=["Segment","Records","Mean score","Median score","Std. deviation"]
    heading("Segment Performance Summary"); st.dataframe(summary.round(3),hide_index=True,use_container_width=True)
    st.plotly_chart(chart(px.box(seg,x="Segment",y=TARGET,color="Segment",title=f"Exam scores by {var} segment"),400),use_container_width=True)
    st.plotly_chart(chart(px.histogram(seg,x=TARGET,color="Segment",barmode="overlay",opacity=.65,nbins=30,title="Score distributions by segment"),400),use_container_width=True)
    heading("Two-Factor Segmentation: Attendance × Study Hours")
    if {"Attendance","Hours_Studied"}.issubset(df.columns):
        two=df.copy(); two["Attendance group"]=np.where(two.Attendance>=two.Attendance.median(),"Higher attendance","Lower attendance")
        two["Study group"]=np.where(two.Hours_Studied>=two.Hours_Studied.median(),"Higher study hours","Lower study hours")
        cross=two.groupby(["Attendance group","Study group"],observed=True)[TARGET].agg(["count","mean","median","std"]).reset_index()
        st.dataframe(cross.round(3),hide_index=True,use_container_width=True)
        pivot=cross.pivot(index="Attendance group",columns="Study group",values="mean")
        st.plotly_chart(chart(px.imshow(pivot,text_auto=".2f",color_continuous_scale="Blues",title="Mean exam score by attendance and study group"),350),use_container_width=True)
    heading("Categorical Group Comparison")
    if categorical:
        cat=st.selectbox("Categorical factor",categorical)
        group=df.groupby(cat,observed=True)[TARGET].agg(["count","mean","median","std"]).reset_index()
        st.dataframe(group.round(3),hide_index=True,use_container_width=True)
        st.plotly_chart(chart(px.bar(group,x=cat,y="mean",color=cat,title=f"Average score by {cat}"),360),use_container_width=True)
    st.caption("Segments use median cut-offs and are descriptive. They do not establish that changing a factor causes score changes.")

# ============================================================
# PAGE 6 - RECOMMENDATIONS & REPORTS
# ============================================================
elif page=="Recommendations & Reports":
    heading("Automatically Generated Analyst Insights")
    for title,text in insights: insight(title,text)
    st.divider()
    heading("Evidence-Based Recommendations")
    st.info("Priority is a screening aid based on association strength. It is not an intervention-effect estimate or a student risk label.")
    if len(recs):
        priority=st.multiselect("Filter by priority",sorted(recs.Priority.unique()),default=sorted(recs.Priority.unique()))
        st.dataframe(recs[recs.Priority.isin(priority)],hide_index=True,use_container_width=True)
    else:
        st.info("No statistically significant numerical associations were found for recommendation generation.")
    st.divider()
    heading("Suggested Institutional Action Plan")
    for title,text in [
        ("1. Review attendance patterns","Compare attendance and score distributions. Investigate barriers before proposing attendance initiatives."),
        ("2. Examine study habits","Review study-hour patterns and consider evaluating structured study support."),
        ("3. Consider prior performance","Use previous scores as one possible signal for offering support, not as a fixed label of ability."),
        ("4. Evaluate support programmes","If tutoring or resource initiatives are introduced, collect real outcome data and evaluate results."),
        ("5. Monitor outcomes","Track future attendance, study habits and assessment results to assess whether actions are associated with improvement.")
    ]: insight(title,text)
    st.divider()
    heading("Downloadable Analyst Report")
    report=[
        "STUDENT ACADEMIC PERFORMANCE INTELLIGENCE REPORT",
        f"Records analysed: {len(df):,}",f"Attributes: {df.shape[1]}",
        f"Mean exam score: {df[TARGET].mean():.3f}",f"Median exam score: {df[TARGET].median():.3f}",
        f"Average attendance: {df.Attendance.mean():.3f}" if "Attendance" in df else "",
        f"Average study hours: {df.Hours_Studied.mean():.3f}" if "Hours_Studied" in df else "",
        "", "CORRELATION RESULTS"
    ]
    for _,r in corr.iterrows():
        report.append(f"{r['Variable']}: Pearson r={r['Pearson r']:.4f}; raw p={r['Pearson p']:.6g}; Holm p={r['Holm adjusted p']:.6g}")
    report += ["","INSIGHTS"]
    for title,text in insights: report += [title,text,""]
    report += ["RECOMMENDATIONS"]
    for _,r in recs.iterrows(): report += [f"{r['Factor']} [{r['Priority']}]",r["Evidence"],r["Interpretation"],r["Suggested action"],""]
    report += ["LIMITATIONS","The dataset is synthetic. Results should not be generalized to real student populations.",
               "Associations do not establish causation. Recommendations are suggestions for investigation, not proven interventions.",
               "Preprocessing should be checked against the original notebook."]
    st.download_button("Download Analyst Report (TXT)", "\n".join(str(x) for x in report).encode("utf-8"),
                       "student_performance_analyst_report.txt","text/plain",use_container_width=True)
    st.download_button("Download Recommendations (CSV)",recs.to_csv(index=False).encode("utf-8"),
                       "academic_recommendations.csv","text/csv",use_container_width=True)
    st.download_button("Download Correlation Results (CSV)",corr.to_csv(index=False).encode("utf-8"),
                       "correlation_analysis.csv","text/csv",use_container_width=True)
    st.download_button("Download Cleaned Dataset (CSV)",df.to_csv(index=False).encode("utf-8"),
                       "cleaned_student_performance.csv","text/csv",use_container_width=True)
    st.divider()
    heading("Limitations and Ethical Considerations")
    st.markdown("""
    - The dataset is synthetic and is not a verified sample of real students.
    - Correlation and group differences do not establish causation.
    - Statistical significance should be considered alongside effect size and sample size.
    - Recommendations are hypotheses for further investigation, not proven interventions.
    - Descriptive segments must not be used to label or penalize individual students.
    - Verify that preprocessing matches the original Kaggle notebook before formal reporting.
    """)

st.divider()
st.markdown('<div style="text-align:center;color:#64748B;font-size:12px;padding:15px 0;">Student Academic Performance Intelligence & Decision Support System<br>Computing for Data Science | Statistical Analysis Project</div>',unsafe_allow_html=True)
