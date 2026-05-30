# =============================================================================
# Unemployment Analysis with Python
# Author: Fahim Ahmed | BUET Urban & Regional Planning
# Dataset: Unemployment in India (May 2019 – June 2020)
# =============================================================================

# --- SECTION 1: IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
print("✅ All libraries imported successfully.")

# =============================================================================
# --- SECTION 2: LOAD THE DATA ---
# =============================================================================

df = pd.read_csv("Unemployment.csv")
# pd.read_csv() reads a CSV file into a DataFrame (a table-like structure).

print("\n📊 RAW DATASET OVERVIEW")
print("=" * 50)
print(f"Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

# =============================================================================
# --- SECTION 3: DATA CLEANING ---
# =============================================================================

print("\n🧹 CLEANING DATA...")

# Step 3a: Drop completely empty rows (the CSV has blank separator rows)
df = df.dropna(subset=['Region'])
# dropna(subset=['Region']) removes any row where Region is empty.
# Those 28 blank separator rows between Rural/Urban blocks get removed here.

print(f"After removing blank rows: {len(df)} rows")

# Step 3b: Strip leading/trailing whitespace from ALL column names
# The raw CSV has spaces before column names like ' Date', ' Frequency' etc.
df.columns = df.columns.str.strip()
# .str.strip() on columns removes spaces from every column header at once.

# Step 3c: Strip whitespace from text columns
df['Region'] = df['Region'].str.strip()
df['Date'] = df['Date'].str.strip()
df['Frequency'] = df['Frequency'].str.strip()
df['Area'] = df['Area'].str.strip()

# Step 3d: Convert Date from string "31-05-2019" to a proper datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
# format='%d-%m-%Y' tells pandas: day-month-year separated by dashes.
# Without specifying format, pandas might misread day/month positions.

# Step 3e: Rename numeric columns for easier typing
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)
# inplace=True modifies df directly instead of returning a new DataFrame.

# Step 3f: Add a Covid flag column
# India's national lockdown began 25 March 2020. April 2020 onward = full impact.
df['Covid_Period'] = (df['Date'] >= '2020-04-01').astype(int)
# (df['Date'] >= '2020-04-01') creates a column of True/False values.
# .astype(int) converts True→1, False→0.

# Step 3g: Add a readable Month-Year label for plotting
df['Month_Year'] = df['Date'].dt.strftime('%b %Y')
# dt.strftime('%b %Y') formats datetime as "May 2019", "Apr 2020", etc.

print(f"✅ Final clean shape: {df.shape}")
print(f"Date range: {df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}")
print(f"States: {df['Region'].nunique()} | Areas: {df['Area'].unique()}")

# =============================================================================
# --- SECTION 4: EXPLORATORY DATA ANALYSIS ---
# =============================================================================

print("\n📈 EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# --- 4a: Basic statistics ---
print("\nOverall Unemployment Rate Statistics:")
print(df['Unemployment_Rate'].describe().round(2))
# .describe() gives count, mean, std, min, 25th percentile, median, 75th, max.

# --- 4b: Pre-Covid vs Covid overall averages ---
pre = df[df['Covid_Period'] == 0]['Unemployment_Rate']
covid = df[df['Covid_Period'] == 1]['Unemployment_Rate']
# df[condition] filters rows. We filter by Covid_Period value.

print(f"\n📊 Pre-Covid Average (May 2019 – Mar 2020):  {pre.mean():.2f}%")
print(f"📊 Covid Period Average (Apr – Jun 2020):     {covid.mean():.2f}%")
print(f"📊 Absolute Increase:                        +{covid.mean() - pre.mean():.2f} percentage points")
print(f"📊 Relative Increase:                        +{((covid.mean()-pre.mean())/pre.mean()*100):.1f}%")

# --- 4c: Lockdown month specifically (April 2020) ---
lockdown_april = df[df['Date'] == '2020-04-30']['Unemployment_Rate']
print(f"\n📊 April 2020 (first full lockdown month): {lockdown_april.mean():.2f}%")

# --- 4d: Per-state averages during covid ---
state_covid = (df[df['Covid_Period'] == 1]
               .groupby('Region')['Unemployment_Rate']
               .mean()
               .sort_values(ascending=False)
               .round(2))
# .groupby('Region') groups all rows by state name.
# .mean() calculates average per group.
# .sort_values(ascending=False) puts highest first.

print("\n🔴 Top 10 States — Highest Unemployment During Covid:")
print(state_covid.head(10).to_string())

print("\n🟢 Top 5 States — Lowest Unemployment During Covid:")
print(state_covid.tail(5).to_string())

# --- 4e: Rural vs Urban breakdown ---
area_table = df.groupby(['Area', 'Covid_Period'])['Unemployment_Rate'].mean().round(2).unstack()
# .unstack() pivots Covid_Period (0 and 1) into separate columns.
area_table.columns = ['Pre-Covid', 'During Covid']
print("\n🏘 Rural vs Urban Unemployment:")
print(area_table)

# --- 4f: Peak unemployment month nationally ---
monthly_nat = df.groupby('Date')['Unemployment_Rate'].mean()
peak_date = monthly_nat.idxmax()
# .idxmax() returns the INDEX (date) where value is highest.
print(f"\n📍 Peak national unemployment: {monthly_nat[peak_date]:.2f}% in {peak_date.strftime('%B %Y')}")

# =============================================================================
# --- SECTION 5: VISUALIZATIONS ---
# =============================================================================

print("\n🎨 Creating visualizations...")

# -----------------------------------------------------------------------
# FIGURE 1: Main 6-panel analysis dashboard
# -----------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
# plt.subplots(rows, cols) creates a grid of subplots.
# figsize=(width_inches, height_inches).

fig.suptitle(
    'India Unemployment Analysis: May 2019 – June 2020\n'
    'CodeAlpha Data Science Internship — Task 2 | Fahim Ahmed, BUET',
    fontsize=13, fontweight='bold', y=1.01
)

# ---- PLOT 1: National unemployment over time (line chart) ----
ax1 = axes[0, 0]  # Top-left cell

monthly_avg = df.groupby('Date')['Unemployment_Rate'].mean().reset_index()
# .reset_index() turns the grouped 'Date' index back into a regular column,
# so we can access it as monthly_avg['Date'].

ax1.plot(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
         color='#2c3e50', linewidth=2.5, marker='o', markersize=5, zorder=3)
# marker='o' adds a dot at each data point.
# zorder=3 puts the line on top of the shaded regions.

# Shade the Covid period with a light red background
ax1.axvspan(pd.Timestamp('2020-04-01'), monthly_avg['Date'].max(),
            alpha=0.15, color='red', label='Covid Period')
# axvspan(x_start, x_end) fills a vertical band between two x values.

# Vertical dashed line at lockdown start
ax1.axvline(pd.Timestamp('2020-04-01'), color='red', linestyle='--',
            linewidth=1.5, label='Lockdown Begins (Apr 2020)', zorder=4)
# axvline() draws a vertical line at one specific x position.

ax1.fill_between(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
                 alpha=0.1, color='#2c3e50')
# fill_between() fills the area under the line. alpha=0.1 = 90% transparent.

ax1.set_title('National Average Unemployment Rate Over Time', fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Unemployment Rate (%)')
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# DateFormatter controls how tick labels appear. '%b %Y' = "May 2019".
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
# setp() sets properties on a list of objects. Rotates labels 45° and aligns right.

# ---- PLOT 2: Pre-Covid vs Covid bar chart ----
ax2 = axes[0, 1]  # Top-right

categories = ['Pre-Covid\n(May 2019 – Mar 2020)', 'During Covid\n(Apr – Jun 2020)']
values = [pre.mean(), covid.mean()]
colors = ['#27ae60', '#e74c3c']  # Green for pre, red for covid

bars = ax2.bar(categories, values, color=colors, edgecolor='black',
               linewidth=0.8, width=0.5)

# Add percentage labels on top of each bar
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f'{val:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=13)
    # bar.get_x() + bar.get_width()/2 = horizontal center of bar.
    # bar.get_height() = top of bar. +0.3 adds a small gap above.

ax2.set_title('Pre-Covid vs Covid: Average Unemployment', fontweight='bold')
ax2.set_ylabel('Average Unemployment Rate (%)')
ax2.set_ylim(0, max(values) * 1.35)
# set_ylim() fixes the y-axis range. *1.35 gives 35% headroom above tallest bar.

# Annotate the change
change = covid.mean() - pre.mean()
ax2.annotate(f'▲ +{change:.1f}pp\n({change/pre.mean()*100:.0f}% increase)',
             xy=(1, covid.mean()), xytext=(0.5, covid.mean() * 0.75),
             fontsize=11, color='#c0392b', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
# annotate() draws text with an optional arrow pointing to (xy).
# xytext is where the text label sits, xy is where the arrow points.

# ---- PLOT 3: Top 10 most affected states (horizontal bar) ----
ax3 = axes[1, 0]

top10 = state_covid.head(10).sort_values(ascending=True)
# We sort ascending=True because barh() flips the order visually —
# the last item in the list ends up at the top of the chart.

# Color bars by severity: darkest red for highest unemployment
norm_vals = (top10.values - top10.min()) / (top10.max() - top10.min())
# Normalize values to 0-1 range for color mapping.
bar_colors = [plt.cm.RdYlGn_r(v) for v in norm_vals]
# RdYlGn_r colormap: 0=green, 1=dark red. _r means reversed.

bars3 = ax3.barh(top10.index, top10.values, color=bar_colors,
                 edgecolor='black', linewidth=0.5)
# barh() = horizontal bar chart. index goes on y-axis, values on x-axis.

for bar, val in zip(bars3, top10.values):
    ax3.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    # bar.get_y() + bar.get_height()/2 = vertical center of bar.

ax3.axvline(pre.mean(), color='green', linestyle='--', linewidth=1.5,
            label=f'Pre-Covid National Avg ({pre.mean():.1f}%)')
ax3.set_title('Top 10 States — Highest Unemployment During Covid', fontweight='bold')
ax3.set_xlabel('Average Unemployment Rate (%) — Apr to Jun 2020')
ax3.legend(fontsize=8)

# ---- PLOT 4: Rural vs Urban grouped bar chart ----
ax4 = axes[1, 1]

rural_pre = df[(df['Area'] == 'Rural') & (df['Covid_Period'] == 0)]['Unemployment_Rate'].mean()
rural_covid = df[(df['Area'] == 'Rural') & (df['Covid_Period'] == 1)]['Unemployment_Rate'].mean()
urban_pre = df[(df['Area'] == 'Urban') & (df['Covid_Period'] == 0)]['Unemployment_Rate'].mean()
urban_covid = df[(df['Area'] == 'Urban') & (df['Covid_Period'] == 1)]['Unemployment_Rate'].mean()
# We filter df by two conditions using & (AND). Both conditions must be True.
# Conditions are wrapped in () because of Python operator precedence rules.

labels = ['Rural', 'Urban']
pre_vals = [rural_pre, urban_pre]
covid_vals = [rural_covid, urban_covid]

x = np.arange(len(labels))   # [0, 1]
width = 0.35                   # width of each individual bar

bars_pre = ax4.bar(x - width/2, pre_vals, width,
                   label='Pre-Covid', color='#3498db', edgecolor='black', linewidth=0.8)
bars_cov = ax4.bar(x + width/2, covid_vals, width,
                   label='During Covid', color='#e74c3c', edgecolor='black', linewidth=0.8)
# x - width/2 shifts pre-covid bars left of center.
# x + width/2 shifts covid bars right of center.
# This creates the side-by-side (grouped) effect.

for bars, vals in [(bars_pre, pre_vals), (bars_cov, covid_vals)]:
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

ax4.set_title('Rural vs Urban: Pre-Covid vs Covid Unemployment', fontweight='bold')
ax4.set_ylabel('Average Unemployment Rate (%)')
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=12)
ax4.legend()
ax4.set_ylim(0, max(covid_vals) * 1.3)

# ---- PLOT 5: Monthly trend by Area (Rural vs Urban line chart) ----
ax5 = axes[2, 0]

for area, color, ls in [('Rural', '#27ae60', '-'), ('Urban', '#8e44ad', '--')]:
    area_monthly = (df[df['Area'] == area]
                    .groupby('Date')['Unemployment_Rate']
                    .mean()
                    .reset_index())
    ax5.plot(area_monthly['Date'], area_monthly['Unemployment_Rate'],
             color=color, linewidth=2, linestyle=ls, marker='o',
             markersize=4, label=area)
# We loop through both Area types and plot one line each.
# ls='-' means solid line, '--' means dashed.

ax5.axvspan(pd.Timestamp('2020-04-01'), df['Date'].max(),
            alpha=0.1, color='red')
ax5.axvline(pd.Timestamp('2020-04-01'), color='red', linestyle='--',
            linewidth=1.2, label='Lockdown (Apr 2020)')

ax5.set_title('Monthly Unemployment: Rural vs Urban Trend', fontweight='bold')
ax5.set_xlabel('Month')
ax5.set_ylabel('Unemployment Rate (%)')
ax5.legend()
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ---- PLOT 6: Distribution histogram — pre-covid vs covid ----
ax6 = axes[2, 1]

ax6.hist(pre.values, bins=30, alpha=0.65, color='#3498db',
         label=f'Pre-Covid (n={len(pre)})', edgecolor='white', linewidth=0.5)
ax6.hist(covid.values, bins=20, alpha=0.65, color='#e74c3c',
         label=f'During Covid (n={len(covid)})', edgecolor='white', linewidth=0.5)
# bins=30 divides the data range into 30 equal-width buckets.
# alpha=0.65 makes bars semi-transparent so both histograms are visible.

ax6.axvline(pre.mean(), color='#2980b9', linestyle='--', linewidth=2,
            label=f'Pre-Covid Mean: {pre.mean():.1f}%')
ax6.axvline(covid.mean(), color='#c0392b', linestyle='--', linewidth=2,
            label=f'Covid Mean: {covid.mean():.1f}%')

ax6.set_title('Distribution of Unemployment Rates\n(Pre-Covid vs Covid)', fontweight='bold')
ax6.set_xlabel('Unemployment Rate (%)')
ax6.set_ylabel('Number of State-Month Observations')
ax6.legend(fontsize=8)

plt.tight_layout()
# tight_layout() automatically adjusts spacing so subplot labels don't overlap.

plt.savefig('unemployment_analysis.png', dpi=150, bbox_inches='tight')
# dpi=150: high resolution (dots per inch). bbox_inches='tight': don't clip labels.
print("✅ Main analysis plot saved.")
plt.close()

# -----------------------------------------------------------------------
# FIGURE 2: State-wise heatmap
# -----------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(16, 11))

# Pivot table: rows=States, columns=Months, values=avg unemployment rate
pivot = df.pivot_table(
    index='Region',
    columns='Date',
    values='Unemployment_Rate',
    aggfunc='mean'
)
# pivot_table reshapes data from "long format" to "wide format".
# Long: one row per (state, month). Wide: one row per state, one column per month.
# aggfunc='mean' averages Rural+Urban together for each state+month cell.

# Format column labels as "May 2019" instead of raw timestamps
pivot.columns = [d.strftime('%b %Y') for d in pivot.columns]
# List comprehension: applies strftime to every column name.

# Sort states by their overall average unemployment (highest at top)
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
# pivot.mean(axis=1): row-wise average (average across all months per state).
# .sort_values(ascending=False): sort states highest→lowest.
# .index: get the sorted state names list.
# .loc[...]: reorder pivot rows in that order.

sns.heatmap(pivot,
            cmap='RdYlGn_r',          # Red=high unemployment, Green=low
            ax=ax,
            linewidths=0.4,           # thin grid lines between cells
            linecolor='white',
            cbar_kws={'label': 'Unemployment Rate (%)', 'shrink': 0.8},
            annot=False,              # don't print numbers in each cell (too crowded)
            fmt='.1f')
# cmap='RdYlGn_r': Red-Yellow-Green reversed.
#   High values = Red (bad = high unemployment).
#   Low values = Green (good = low unemployment).

# Add a vertical line separating pre-covid from covid columns
covid_start_idx = list(pivot.columns).index('Apr 2020')
ax.axvline(x=covid_start_idx, color='red', linewidth=2.5, linestyle='--')
# We find the column index of 'Apr 2020' and draw a red line there.

ax.text(covid_start_idx + 0.1, -0.5, '◀ Pre-Covid | Covid ▶',
        color='red', fontsize=10, fontweight='bold', va='top')
# ax.text(x, y, string) adds text at position (x, y) in axis coordinates.
# y=-0.5 puts it just above the top of the heatmap.

ax.set_title(
    'State-wise Monthly Unemployment Rate Heatmap\n'
    'Red = High | Green = Low | States sorted by average unemployment',
    fontsize=13, fontweight='bold', pad=15
)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('State/Region', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

plt.savefig('unemployment_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ Heatmap saved.")
plt.close()

# -----------------------------------------------------------------------
# FIGURE 3: State-wise Covid impact (change in unemployment)
# -----------------------------------------------------------------------
fig3, ax = plt.subplots(figsize=(14, 9))

# Calculate the CHANGE in unemployment rate for each state
state_pre = df[df['Covid_Period'] == 0].groupby('Region')['Unemployment_Rate'].mean()
state_cov = df[df['Covid_Period'] == 1].groupby('Region')['Unemployment_Rate'].mean()
state_change = (state_cov - state_pre).sort_values(ascending=True)
# We subtract pre-covid average from covid average for each state.
# Positive = worsened, negative = improved (very rare).

colors = ['#e74c3c' if v > 0 else '#27ae60' for v in state_change.values]
# List comprehension: red if unemployment went up, green if it went down.

bars = ax.barh(state_change.index, state_change.values,
               color=colors, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, state_change.values):
    xpos = val + 0.2 if val >= 0 else val - 0.2
    align = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'+{val:.1f}pp' if val >= 0 else f'{val:.1f}pp',
            va='center', ha=align, fontsize=9)
# We put the label to the right of positive bars, left of negative bars.

ax.axvline(0, color='black', linewidth=1)
# Vertical line at 0 — baseline reference.

red_patch = mpatches.Patch(color='#e74c3c', label='Unemployment Increased')
green_patch = mpatches.Patch(color='#27ae60', label='Unemployment Decreased')
ax.legend(handles=[red_patch, green_patch], fontsize=10)
# mpatches.Patch creates a colored rectangle for the legend.

ax.set_title(
    'Change in Unemployment Rate: Pre-Covid → Covid Period\n'
    '(Percentage Point Difference per State)',
    fontsize=13, fontweight='bold'
)
ax.set_xlabel('Change in Unemployment Rate (percentage points)', fontsize=11)
ax.set_ylabel('State/Region', fontsize=11)
plt.tight_layout()
plt.savefig('unemployment_covid_impact.png', dpi=150, bbox_inches='tight')
print("✅ Covid impact chart saved.")
plt.close()

# =============================================================================
# --- SECTION 6: FINAL SUMMARY ---
# =============================================================================

print("\n" + "=" * 60)
print("💡 KEY FINDINGS & INSIGHTS")
print("=" * 60)

worst = state_covid.idxmax()
best = state_covid.idxmin()
most_impacted = state_change.idxmax()
least_impacted = state_change.idxmin()

print(f"""
DATASET:
  • 28 Indian states/UTs | May 2019 – June 2020
  • 740 observations (Rural + Urban, monthly)

FINDING 1 — Pre-Covid Baseline (May 2019 – Mar 2020):
  • National average unemployment: {pre.mean():.2f}%
  • Rural average: {rural_pre:.2f}% | Urban average: {urban_pre:.2f}%

FINDING 2 — Covid-19 Shock (Apr – Jun 2020):
  • National average jumped to: {covid.mean():.2f}%
  • Absolute increase: +{covid.mean()-pre.mean():.2f} percentage points
  • Relative increase: +{(covid.mean()-pre.mean())/pre.mean()*100:.0f}%
  • Peak national month: {peak_date.strftime('%B %Y')} at {monthly_nat[peak_date]:.2f}%

FINDING 3 — Most & Least Affected States:
  • Highest unemployment during Covid: {worst} ({state_covid[worst]:.1f}%)
  • Lowest unemployment during Covid: {best} ({state_covid[best]:.1f}%)
  • Biggest surge from pre→covid: {most_impacted} (+{state_change[most_impacted]:.1f}pp)
  • Most resilient state: {least_impacted} ({state_change[least_impacted]:.1f}pp change)

FINDING 4 — Rural vs Urban:
  • Rural Covid avg: {rural_covid:.2f}% | Urban Covid avg: {urban_covid:.2f}%
  • {'Urban areas were harder hit.' if urban_covid > rural_covid else 'Rural areas were harder hit.'}
  • Reason: Urban economies depend heavily on service sector, retail, 
    transport — all severely disrupted by lockdown.

POLICY RECOMMENDATIONS:
  1. MGNREGA expansion to absorb reverse migrant labor from cities.
  2. State-specific recovery funds for {worst} and other high-unemployment states.
  3. Unemployment insurance scheme for informal urban workers.
  4. Real-time employment monitoring for faster crisis response.
""")

print("✅ FILES PRODUCED:")
print("   unemployment_analysis.png    — 6-panel main dashboard")
print("   unemployment_heatmap.png     — State × Month heatmap")
print("   unemployment_covid_impact.png — Per-state Covid impact chart")
