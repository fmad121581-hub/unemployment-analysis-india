# Unemployment Analysis — India (2019–2020)

An exploratory data analysis of unemployment trends across 28 Indian states from May 2019 to June 2020, with a focus on the economic shock caused by the COVID-19 national lockdown in April 2020.

## Key Findings

- **Pre-COVID national average:** ~8.5% unemployment rate
- **COVID period average (Apr–Jun 2020):** spiked dramatically — over 23% nationally
- **April 2020** (first full lockdown month) saw the most severe unemployment spike
- **Urban areas** were harder hit than rural areas, due to dependence on the service sector and retail
- **Bihar, Jharkhand, and Puducherry** recorded the highest unemployment rates during the lockdown
- **Maharashtra** and other manufacturing hubs saw severe urban unemployment

## Output Charts

| File | Description |
|---|---|
| `unemployment_analysis.png` | 6-panel main dashboard (trends, distributions, rural vs urban) |
| `unemployment_heatmap.png` | State × Month heatmap showing regional variation |
| `unemployment_covid_impact.png` | Per-state bar chart of COVID impact (pre vs during) |

## Project Structure

```
unemployment-analysis-india/
│
├── unemployment_analysis.py         # Main analysis script
├── Unemployment in India.csv        # Dataset (740 rows, 28 states, Rural + Urban)
├── unemployment_analysis.png
├── unemployment_heatmap.png
├── unemployment_covid_impact.png
└── README.md
```

## Dataset

- **Source:** Centre for Monitoring Indian Economy (CMIE)
- **Coverage:** 28 Indian states/UTs, May 2019 – June 2020
- **Rows:** 740 observations (Rural + Urban, monthly)
- **Columns:** Region, Date, Unemployment Rate (%), Employed, Labour Participation Rate (%), Area

## What the Script Does

1. **Data Cleaning** — strips whitespace from column names, parses dates, renames columns
2. **COVID Flag** — adds a binary column marking April 2020 onward as the COVID period
3. **EDA** — national trend, state-level breakdown, rural vs urban comparison
4. **Visualizations** — 6-panel dashboard, heatmap, per-state COVID impact chart
5. **Policy Insights** — data-driven recommendations for employment recovery

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn

# Place the dataset in the same folder as the script, then run:
python unemployment_analysis.py
```

## Policy Recommendations (from analysis)

1. MGNREGA expansion to absorb reverse-migrant labor from cities
2. State-specific recovery funds for highest-unemployment states
3. Unemployment insurance for informal urban workers
4. Real-time employment monitoring for faster crisis response

## Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn

## Author

**Fahim Ahmed**  
2nd Year Student, Urban & Regional Planning  
Bangladesh University of Engineering and Technology (BUET)  
[LinkedIn](https://www.linkedin.com/in/fahim-ahmed-585b26357) | [GitHub](https://github.com/fmad121581-hub)
