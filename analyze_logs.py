import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def analyze_logs(filepath='logs_Main.csv'):
    df = pd.read_csv(filepath)
    
    if df['Human_Override'].dtype == 'object':
        df['Human_Override'] = df['Human_Override'].astype(str).str.strip().str.upper() == 'TRUE'
        
    df['Human_Override_Int'] = df['Human_Override'].astype(int)
    df = df.dropna(subset=['Trust_Score_ESM', 'Professional_Autonomy_ESM'])
    df['Trust_Score_ESM'] = pd.to_numeric(df['Trust_Score_ESM'], errors='coerce')
    df['Professional_Autonomy_ESM'] = pd.to_numeric(df['Professional_Autonomy_ESM'], errors='coerce')
    df = df.dropna(subset=['Trust_Score_ESM', 'Professional_Autonomy_ESM'])

    print(f"Total valid log entries analyzed: {len(df)}")
    
    override_rates = df.groupby('Agent_System')['Human_Override'].mean() * 100
    print("\n--- Override Rates by System ---")
    print(override_rates.to_string(float_format="%.2f%%"))

    print("\n--- Mean ESM Scores: Override vs. No Override ---")
    esm_means = df.groupby('Human_Override')[['Trust_Score_ESM', 'Professional_Autonomy_ESM']].mean()
    print(esm_means)
    
    overrides = df[df['Human_Override'] == True]
    no_overrides = df[df['Human_Override'] == False]
    
    if len(overrides) > 1 and len(no_overrides) > 1:
        stat_lev, p_lev = stats.levene(no_overrides['Trust_Score_ESM'], overrides['Trust_Score_ESM'])
        print("\n--- Robustness Check: Levene's Test for Equality of Variances ---")
        print(f"Statistic = {stat_lev:.3f}, p-value = {p_lev:.4f}")
        
        equal_var_assumption = p_lev > 0.05
        t_trust, p_trust = stats.ttest_ind(no_overrides['Trust_Score_ESM'], overrides['Trust_Score_ESM'], equal_var=equal_var_assumption)
        
        print("\n--- T-Test Results ---")
        print(f"Trust Score: t = {t_trust:.3f}, p = {p_trust:.4f} (Equal Var Assumed: {equal_var_assumption})")
        
        def cohend(d1, d2):
            n1, n2 = len(d1), len(d2)
            s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
            s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            return (np.mean(d1) - np.mean(d2)) / s
            
        print(f"Cohen's d (Trust): {cohend(no_overrides['Trust_Score_ESM'], overrides['Trust_Score_ESM']):.3f}")

    print("\n--- A* Robust Regression Model (OLS with HC3 Robust Standard Errors) ---")
    try:
        model = ols('Trust_Score_ESM ~ Human_Override_Int + Professional_Autonomy_ESM + C(Agent_System)', data=df).fit(cov_type='HC3')
        print(model.summary().tables[1])
        print(f"R-squared: {model.rsquared:.3f}, F-statistic p-value: {model.f_pvalue:.4f}")
        
        print("\n--- Robustness Check: Multicollinearity (VIF) ---")
        y, X = dmatrices('Trust_Score_ESM ~ Human_Override_Int + Professional_Autonomy_ESM + C(Agent_System)', data=df, return_type='dataframe')
        vif = pd.DataFrame()
        vif['Variable'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif.to_string(index=False))
        
    except Exception as e:
        print(f"Regression could not be run: {e}")

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='Human_Override', y='Trust_Score_ESM', hue='Human_Override', data=df, palette='Set2', legend=False)
    sns.stripplot(x='Human_Override', y='Trust_Score_ESM', hue='Human_Override', data=df, color=".25", alpha=0.5, legend=False)
    # plt.title('Relational Dissonance: ESM Trust Scores during Autonomous AI Overrides', fontsize=14)
    plt.xlabel('Human Override Executed', fontsize=12)
    plt.ylabel('Trust Score (1-5)', fontsize=12)
    plt.xticks([0, 1], ['False (AI Accepted)', 'True (Human Override)'])
    plt.tight_layout()
    plt.savefig('fig1_trust_override_boxplot.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=override_rates.index, y=override_rates.values, hue=override_rates.index, palette='Blues_d', legend=False)
    # plt.title('Frequency of Critical Incidents (Override Rates) by AI System', fontsize=14)
    plt.ylabel('Override Rate (%)', fontsize=12)
    plt.ylim(0, max(override_rates.values) + 10 if len(override_rates) > 0 else 100)
    plt.tight_layout()
    plt.savefig('fig2_override_rates.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    analyze_logs()