import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import os

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2E4057', '#048A81', '#8B5CF6', '#F59E0B', '#EF4444', '#3B82F6']
os.makedirs('charts', exist_ok=True)

def save_chart(fig, name):
    fig.savefig(f'charts/{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved charts/{name}.png")

# Load data
df = pd.read_csv('data/cookie_cats_ab_test.csv')

print("=" * 65)
print("A/B TEST ANALYSIS: COOKIE CATS RETENTION EXPERIMENT")
print("=" * 65)

# ============================================================
# 1. EXPERIMENT OVERVIEW
# ============================================================
print("\n1. EXPERIMENT OVERVIEW")
print("-" * 45)
print("  Product: Cookie Cats (mobile puzzle game)")
print("  Test: Moving the first gate from level 30 to level 40")
print("  Control (gate_30): Gate appears at level 30")
print("  Treatment (gate_40): Gate appears at level 40")
print("  Hypothesis: Moving the gate later will improve retention")
print("     by letting players build more engagement before")
print("     hitting a forced pause.")
print(f"  Total users in experiment: {len(df):,}")

# ============================================================
# 2. DATA VALIDATION AND BALANCE CHECK
# ============================================================
print("\n2. DATA VALIDATION AND BALANCE CHECK")
print("-" * 45)

group_sizes = df['version'].value_counts()
print(f"  Control (gate_30):   {group_sizes['gate_30']:,} users")
print(f"  Treatment (gate_40): {group_sizes['gate_40']:,} users")
print(f"  Split ratio: {group_sizes['gate_30'] / len(df) * 100:.1f}% / {group_sizes['gate_40'] / len(df) * 100:.1f}%")

balance = group_sizes['gate_30'] / group_sizes['gate_40']
if 0.95 <= balance <= 1.05:
    print("  Balance check: PASS (groups are roughly equal)")
else:
    print(f"  Balance check: WARNING (ratio is {balance:.2f}, expected ~1.0)")

print(f"\n  Missing values:")
print(f"    retention_1: {df['retention_1'].isna().sum()}")
print(f"    retention_7: {df['retention_7'].isna().sum()}")
print(f"    sum_gamerounds: {df['sum_gamerounds'].isna().sum()}")

# Balance chart
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['Control\n(gate_30)', 'Treatment\n(gate_40)'],
              [group_sizes['gate_30'], group_sizes['gate_40']],
              color=[COLORS[0], COLORS[1]], alpha=0.85, edgecolor='white', width=0.5)
ax.set_ylabel('Number of Users', fontsize=12)
ax.set_title('Experiment Group Sizes', fontsize=14, fontweight='bold', pad=15)
for bar, val in zip(bars, [group_sizes['gate_30'], group_sizes['gate_40']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
save_chart(fig, '01_group_balance')

# ============================================================
# 3. DAY 1 RETENTION ANALYSIS
# ============================================================
print("\n3. DAY 1 RETENTION ANALYSIS")
print("-" * 45)

control = df[df['version'] == 'gate_30']
treatment = df[df['version'] == 'gate_40']

d1_control = control['retention_1'].mean()
d1_treatment = treatment['retention_1'].mean()
d1_diff = d1_control - d1_treatment
d1_lift = (d1_control - d1_treatment) / d1_treatment * 100

print(f"  Control (gate_30)  D1 retention: {d1_control:.4f} ({d1_control*100:.2f}%)")
print(f"  Treatment (gate_40) D1 retention: {d1_treatment:.4f} ({d1_treatment*100:.2f}%)")
print(f"  Absolute difference: {d1_diff:.4f} ({d1_diff*100:.2f} pp)")
print(f"  Relative lift: {d1_lift:+.2f}%")

# Z-test for D1
n_c1 = len(control)
n_t1 = len(treatment)
p_c1 = d1_control
p_t1 = d1_treatment
p_pooled_1 = (p_c1 * n_c1 + p_t1 * n_t1) / (n_c1 + n_t1)
se_1 = np.sqrt(p_pooled_1 * (1 - p_pooled_1) * (1/n_c1 + 1/n_t1))
z_score_1 = (p_c1 - p_t1) / se_1
p_value_1 = 2 * (1 - stats.norm.cdf(abs(z_score_1)))

# Confidence interval
se_diff_1 = np.sqrt(p_c1 * (1-p_c1) / n_c1 + p_t1 * (1-p_t1) / n_t1)
ci_lower_1 = (p_c1 - p_t1) - 1.96 * se_diff_1
ci_upper_1 = (p_c1 - p_t1) + 1.96 * se_diff_1

print(f"\n  Statistical Test (two-proportion z-test):")
print(f"    Z-score: {z_score_1:.4f}")
print(f"    P-value: {p_value_1:.4f}")
print(f"    95% CI for difference: [{ci_lower_1*100:.2f}%, {ci_upper_1*100:.2f}%]")
if p_value_1 < 0.05:
    print(f"    Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print(f"    Result: NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")

# ============================================================
# 4. DAY 7 RETENTION ANALYSIS
# ============================================================
print("\n4. DAY 7 RETENTION ANALYSIS")
print("-" * 45)

d7_control = control['retention_7'].mean()
d7_treatment = treatment['retention_7'].mean()
d7_diff = d7_control - d7_treatment
d7_lift = (d7_control - d7_treatment) / d7_treatment * 100

print(f"  Control (gate_30)  D7 retention: {d7_control:.4f} ({d7_control*100:.2f}%)")
print(f"  Treatment (gate_40) D7 retention: {d7_treatment:.4f} ({d7_treatment*100:.2f}%)")
print(f"  Absolute difference: {d7_diff:.4f} ({d7_diff*100:.2f} pp)")
print(f"  Relative lift: {d7_lift:+.2f}%")

# Z-test for D7
p_c7 = d7_control
p_t7 = d7_treatment
p_pooled_7 = (p_c7 * n_c1 + p_t7 * n_t1) / (n_c1 + n_t1)
se_7 = np.sqrt(p_pooled_7 * (1 - p_pooled_7) * (1/n_c1 + 1/n_t1))
z_score_7 = (p_c7 - p_t7) / se_7
p_value_7 = 2 * (1 - stats.norm.cdf(abs(z_score_7)))

se_diff_7 = np.sqrt(p_c7 * (1-p_c7) / n_c1 + p_t7 * (1-p_t7) / n_t1)
ci_lower_7 = (p_c7 - p_t7) - 1.96 * se_diff_7
ci_upper_7 = (p_c7 - p_t7) + 1.96 * se_diff_7

print(f"\n  Statistical Test (two-proportion z-test):")
print(f"    Z-score: {z_score_7:.4f}")
print(f"    P-value: {p_value_7:.4f}")
print(f"    95% CI for difference: [{ci_lower_7*100:.2f}%, {ci_upper_7*100:.2f}%]")
if p_value_7 < 0.05:
    print(f"    Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print(f"    Result: NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")

# Retention comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# D1 retention
bars1 = ax1.bar(['Control\n(gate_30)', 'Treatment\n(gate_40)'],
                [d1_control * 100, d1_treatment * 100],
                color=[COLORS[0], COLORS[1]], alpha=0.85, edgecolor='white', width=0.5)
ax1.set_ylabel('Retention Rate (%)', fontsize=12)
ax1.set_title(f'Day 1 Retention\np-value: {p_value_1:.4f}', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylim(0, 55)
for bar, val in zip(bars1, [d1_control * 100, d1_treatment * 100]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

# D7 retention
bars2 = ax2.bar(['Control\n(gate_30)', 'Treatment\n(gate_40)'],
                [d7_control * 100, d7_treatment * 100],
                color=[COLORS[0], COLORS[1]], alpha=0.85, edgecolor='white', width=0.5)
ax2.set_ylabel('Retention Rate (%)', fontsize=12)
ax2.set_title(f'Day 7 Retention\np-value: {p_value_7:.4f}', fontsize=13, fontweight='bold', pad=15)
ax2.set_ylim(0, 25)
for bar, val in zip(bars2, [d7_control * 100, d7_treatment * 100]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
save_chart(fig, '02_retention_comparison')

# ============================================================
# 5. ENGAGEMENT ANALYSIS
# ============================================================
print("\n5. ENGAGEMENT ANALYSIS (Game Rounds Played)")
print("-" * 45)

eng_control = control['sum_gamerounds'].describe()
eng_treatment = treatment['sum_gamerounds'].describe()

print(f"  Control  - Mean: {eng_control['mean']:.1f}, Median: {eng_control['50%']:.0f}, Std: {eng_control['std']:.1f}")
print(f"  Treatment - Mean: {eng_treatment['mean']:.1f}, Median: {eng_treatment['50%']:.0f}, Std: {eng_treatment['std']:.1f}")

# Mann-Whitney U test (non-parametric, better for skewed game round data)
u_stat, u_pvalue = stats.mannwhitneyu(
    control['sum_gamerounds'],
    treatment['sum_gamerounds'],
    alternative='two-sided'
)
print(f"\n  Mann-Whitney U test (non-parametric):")
print(f"    U-statistic: {u_stat:,.0f}")
print(f"    P-value: {u_pvalue:.4f}")
if u_pvalue < 0.05:
    print(f"    Result: STATISTICALLY SIGNIFICANT difference in engagement")
else:
    print(f"    Result: NO SIGNIFICANT difference in engagement")

# Engagement distribution chart (capped at 100 rounds for readability)
fig, ax = plt.subplots(figsize=(12, 6))
bins = range(0, 101, 2)
ax.hist(control['sum_gamerounds'].clip(upper=100), bins=bins,
        alpha=0.6, label='Control (gate_30)', color=COLORS[0], edgecolor='white')
ax.hist(treatment['sum_gamerounds'].clip(upper=100), bins=bins,
        alpha=0.6, label='Treatment (gate_40)', color=COLORS[1], edgecolor='white')
ax.set_xlabel('Game Rounds Played (capped at 100)', fontsize=12)
ax.set_ylabel('Number of Users', fontsize=12)
ax.set_title('Engagement Distribution: Control vs Treatment', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
plt.tight_layout()
save_chart(fig, '03_engagement_distribution')

# ============================================================
# 6. RETENTION BY ENGAGEMENT LEVEL
# ============================================================
print("\n6. RETENTION BY ENGAGEMENT LEVEL")
print("-" * 45)

df['engagement_level'] = pd.cut(df['sum_gamerounds'],
    bins=[-1, 0, 10, 50, 100, 50000],
    labels=['0 rounds', '1-10', '11-50', '51-100', '100+'])

engagement_retention = df.groupby(['engagement_level', 'version'], observed=True).agg(
    users=('userid', 'count'),
    d1_retained=('retention_1', 'mean'),
    d7_retained=('retention_7', 'mean')
).reset_index()

engagement_retention['d1_retained'] = (engagement_retention['d1_retained'] * 100).round(2)
engagement_retention['d7_retained'] = (engagement_retention['d7_retained'] * 100).round(2)

print(engagement_retention.to_string(index=False))

# Engagement level D7 retention chart
pivot = engagement_retention.pivot(index='engagement_level', columns='version', values='d7_retained').reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(pivot))
width = 0.35
ax.bar([i - width/2 for i in x], pivot['gate_30'], width,
       label='Control (gate_30)', color=COLORS[0], alpha=0.85)
ax.bar([i + width/2 for i in x], pivot['gate_40'], width,
       label='Treatment (gate_40)', color=COLORS[1], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(pivot['engagement_level'])
ax.set_xlabel('Engagement Level (Game Rounds)', fontsize=12)
ax.set_ylabel('Day 7 Retention (%)', fontsize=12)
ax.set_title('Day 7 Retention by Engagement Level and Test Group', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
plt.tight_layout()
save_chart(fig, '04_retention_by_engagement')

# ============================================================
# 7. EFFECT SIZE
# ============================================================
print("\n7. EFFECT SIZE AND PRACTICAL SIGNIFICANCE")
print("-" * 45)

# Cohen's h for proportions
h_d1 = 2 * np.arcsin(np.sqrt(d1_control)) - 2 * np.arcsin(np.sqrt(d1_treatment))
h_d7 = 2 * np.arcsin(np.sqrt(d7_control)) - 2 * np.arcsin(np.sqrt(d7_treatment))

print(f"  Cohen's h (D1): {h_d1:.4f}", end="")
if abs(h_d1) < 0.2:
    print(" (negligible effect)")
elif abs(h_d1) < 0.5:
    print(" (small effect)")
else:
    print(" (medium+ effect)")

print(f"  Cohen's h (D7): {h_d7:.4f}", end="")
if abs(h_d7) < 0.2:
    print(" (negligible effect)")
elif abs(h_d7) < 0.5:
    print(" (small effect)")
else:
    print(" (medium+ effect)")

# Practical significance
d7_users_affected = abs(d7_diff) * len(df)
print(f"\n  Practical impact (D7):")
print(f"    {abs(d7_diff)*100:.2f} percentage point difference")
print(f"    Across {len(df):,} users, that is ~{d7_users_affected:.0f} additional users retained")

# Effect size chart
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['Day 1\nRetention', 'Day 7\nRetention']
effects = [h_d1, h_d7]
bar_colors = [COLORS[3] if abs(e) < 0.2 else COLORS[1] for e in effects]
bars = ax.bar(metrics, effects, color=bar_colors, alpha=0.85, edgecolor='white', width=0.4)
ax.axhline(y=0.2, color=COLORS[4], linestyle='--', alpha=0.5, label='Small effect threshold (0.2)')
ax.axhline(y=-0.2, color=COLORS[4], linestyle='--', alpha=0.5)
ax.set_ylabel("Cohen's h (effect size)", fontsize=12)
ax.set_title("Effect Size: Control vs Treatment", fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)
for bar, val in zip(bars, effects):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.015,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
save_chart(fig, '05_effect_size')

# ============================================================
# 8. SUMMARY AND RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 65)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 65)

print(f"\nExperiment Results:")
print(f"  Day 1 Retention: Control {d1_control*100:.2f}% vs Treatment {d1_treatment*100:.2f}%")
print(f"    Difference: {d1_diff*100:.2f} pp | p-value: {p_value_1:.4f} | Effect: {abs(h_d1):.4f}")
print(f"  Day 7 Retention: Control {d7_control*100:.2f}% vs Treatment {d7_treatment*100:.2f}%")
print(f"    Difference: {d7_diff*100:.2f} pp | p-value: {p_value_7:.4f} | Effect: {abs(h_d7):.4f}")

print(f"\nConclusion:")
if p_value_7 < 0.05 and d7_diff > 0:
    print(f"  The control (gate_30) shows HIGHER Day 7 retention than treatment.")
    print(f"  Moving the gate to level 40 DECREASED retention.")
    print(f"  RECOMMENDATION: Do NOT ship the change. Keep the gate at level 30.")
elif p_value_7 < 0.05 and d7_diff < 0:
    print(f"  The treatment (gate_40) shows HIGHER Day 7 retention than control.")
    print(f"  Moving the gate to level 40 INCREASED retention.")
    print(f"  RECOMMENDATION: Ship the change. Move the gate to level 40.")
else:
    print(f"  The difference is NOT statistically significant at the 0.05 level.")
    print(f"  We cannot confidently say the gate position affects retention.")
    print(f"  RECOMMENDATION: Keep the gate at level 30 (status quo).")

print(f"\nKey Takeaways:")
print(f"  1. Statistical significance alone is not enough. Effect size matters.")
print(f"     A tiny effect on 90K users may be statistically significant but")
print(f"     practically meaningless.")
print(f"  2. Always check both short-term (D1) and long-term (D7) metrics.")
print(f"     A change that looks neutral at D1 may show its impact at D7.")
print(f"  3. Segment the analysis by engagement level. The treatment may")
print(f"     help some users while hurting others.")
print(f"  4. The recommendation should account for practical significance,")
print(f"     not just p-values. Would you change your product based on a")
print(f"     {abs(d7_diff)*100:.2f} percentage point difference?")

print(f"\n{'=' * 65}")
print(f"Analysis complete. 5 charts saved to /charts directory.")
print(f"{'=' * 65}")
