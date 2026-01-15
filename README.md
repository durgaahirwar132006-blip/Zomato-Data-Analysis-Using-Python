# 🍽️ Zomato Restaurant Data Analysis

**Author:** Durga Ahirwar  
**Date:** January 2026  
**Project:** GeeksforGeeks Data Analysis Certification

## 📋 Project Overview
This notebook performs comprehensive analysis on Zomato restaurant data to identify trends in ratings, pricing, customer preferences, and service offerings.

### 🎯 Objectives
1. Analyze restaurant ratings distribution
2. Understand price vs quality relationship
3. Compare online vs offline ordering trends
4. Identify top-performing restaurants
5. Examine customer engagement patterns
## 📚 Import Libraries
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")
## 📊 Load Dataset
# Load the dataset
df = pd.read_csv('../data/zomato_data.csv')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
## 🔍 Initial Data Exploration
# Display first few rows
print("📝 First 5 rows:")
df.head()
# Dataset information
print("ℹ️ Dataset Info:")
df.info()
# Statistical summary
print("📈 Statistical Summary:")
df.describe()
# Check for missing values
print("🔎 Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values!")
# Column names
print("📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")
## 🧹 Data Cleaning & Preparation
# Parse rating from "X.X/5" format to numeric
def parse_rating(rate_str):
    try:
        if isinstance(rate_str, str) and '/' in rate_str:
            return float(rate_str.split('/')[0])
        return float(rate_str)
    except:
        return np.nan

df['rating'] = df['rate'].apply(parse_rating)

print("✅ Rating column cleaned")
print(f"   Numeric ratings: {df['rating'].notna().sum()}")
# Create price categories
df['price_range'] = pd.cut(
    df['approx_cost(for two people)'],
    bins=[0, 300, 500, 700, float('inf')],
    labels=['Budget (<₹300)', 'Affordable (₹300-500)', 
            'Mid-Range (₹500-700)', 'Premium (>₹700)']
)

print("✅ Price range categories created")
print("\nPrice Distribution:")
print(df['price_range'].value_counts())
# Create rating categories
df['rating_category'] = pd.cut(
    df['rating'],
    bins=[0, 3.0, 3.5, 4.0, 5.0],
    labels=['Poor (<3.0)', 'Average (3.0-3.5)', 
            'Good (3.5-4.0)', 'Excellent (>4.0)']
)

print("✅ Rating categories created")
print("\nRating Distribution:")
print(df['rating_category'].value_counts())
## 📊 Exploratory Data Analysis (EDA)

### 1️⃣ Restaurant Overview
# Key statistics
print("🍽️ RESTAURANT STATISTICS")
print("=" * 50)
print(f"Total Restaurants: {len(df)}")
print(f"Average Rating: {df['rating'].mean():.2f}/5.0")
print(f"Median Rating: {df['rating'].median():.2f}/5.0")
print(f"Average Cost for Two: ₹{df['approx_cost(for two people)'].mean():.0f}")
print(f"Total Customer Votes: {df['votes'].sum():,}")
print(f"Restaurant Types: {df['listed_in(type)'].nunique()}")
print(f"\nOnline Ordering Available: {(df['online_order'] == 'Yes').sum()} ({(df['online_order'] == 'Yes').sum() / len(df) * 100:.1f}%)")
print(f"Table Booking Available: {(df['book_table'] == 'Yes').sum()} ({(df['book_table'] == 'Yes').sum() / len(df) * 100:.1f}%)")
### 2️⃣ Online vs Offline Service Analysis
# Online order distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
online_counts = df['online_order'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']
axes[0].pie(online_counts, labels=online_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors, explode=(0.05, 0))
axes[0].set_title('Online Order Availability', fontsize=14, fontweight='bold')

# Bar chart
online_counts.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black', alpha=0.8)
axes[1].set_title('Restaurant Count by Service Type', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_xlabel('Online Order', fontsize=11)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('../visuals/online_order_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visualization saved: online_order_distribution.png")
### 3️⃣ Rating Distribution Analysis
# Rating distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(df['rating'].dropna(), bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['rating'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {df["rating"].mean():.2f}')
axes[0, 0].set_xlabel('Rating', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Rating Distribution', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Box plot
box = axes[0, 1].boxplot(df['rating'].dropna(), patch_artist=True)
box['boxes'][0].set_facecolor('lightcoral')
axes[0, 1].set_ylabel('Rating', fontsize=11)
axes[0, 1].set_title('Rating Box Plot', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Rating categories
rating_cat = df['rating_category'].value_counts().sort_index()
rating_cat.plot(kind='bar', ax=axes[1, 0], color='green', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Restaurants by Rating Category', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_xlabel('Rating Category', fontsize=11)
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].grid(alpha=0.3)

# Rating by online order
online_rating = df.groupby('online_order')['rating'].mean()
online_rating.plot(kind='bar', ax=axes[1, 1], color=['#FF6B6B', '#4ECDC4'], 
                   edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Average Rating by Online Order', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Average Rating', fontsize=11)
axes[1, 1].set_xlabel('Online Order', fontsize=11)
axes[1, 1].tick_params(axis='x', rotation=0)
axes[1, 1].axhline(df['rating'].mean(), color='red', linestyle='--', linewidth=2, label='Overall Avg')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visuals/rating_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visualization saved: rating_distribution.png")
### 4️⃣ Restaurant Type Analysis
# Analyze by restaurant type
type_analysis = df.groupby('listed_in(type)').agg({
    'name': 'count',
    'rating': 'mean',
    'votes': 'sum',
    'approx_cost(for two people)': 'mean'
}).round(2)

type_analysis.columns = ['Count', 'Avg_Rating', 'Total_Votes', 'Avg_Cost']
type_analysis = type_analysis.sort_values('Total_Votes', ascending=False)

print("📊 Restaurant Type Statistics:")
print(type_analysis)
# Visualize restaurant types
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Count by type
type_analysis['Count'].plot(kind='bar', ax=axes[0, 0], color='steelblue', 
                             edgecolor='black', alpha=0.8)
axes[0, 0].set_title('Restaurant Count by Type', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Count', fontsize=11)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(alpha=0.3)

# Total votes
type_analysis['Total_Votes'].plot(kind='bar', ax=axes[0, 1], color='coral',
                                   edgecolor='black', alpha=0.8)
axes[0, 1].set_title('Customer Engagement by Type', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Total Votes', fontsize=11)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(alpha=0.3)

# Average rating
type_analysis['Avg_Rating'].plot(kind='bar', ax=axes[1, 0], color='mediumseagreen',
                                  edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Average Rating by Type', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Average Rating', fontsize=11)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].axhline(df['rating'].mean(), color='red', linestyle='--', linewidth=2)
axes[1, 0].grid(alpha=0.3)

# Average cost
type_analysis['Avg_Cost'].plot(kind='bar', ax=axes[1, 1], color='orange',
                                edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Average Cost by Type', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Average Cost (₹)', fontsize=11)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visuals/restaurant_type_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visualization saved: restaurant_type_analysis.png")
### 5️⃣ Price Range Analysis
# Analyze by price range
price_analysis = df.groupby('price_range').agg({
    'name': 'count',
    'rating': 'mean',
    'votes': 'sum'
}).round(2)

price_analysis.columns = ['Count', 'Avg_Rating', 'Total_Votes']

print("💰 Price Range Statistics:")
print(price_analysis)
# Visualize price analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Count by price
price_analysis['Count'].plot(kind='bar', ax=axes[0], color='purple',
                              edgecolor='black', alpha=0.8)
axes[0].set_title('Restaurants by Price Range', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11)
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(alpha=0.3)

# Total votes by price
price_analysis['Total_Votes'].plot(kind='bar', ax=axes[1], color='crimson',
                                    edgecolor='black', alpha=0.8)
axes[1].set_title('Customer Preference by Price', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Total Votes', fontsize=11)
axes[1].tick_params(axis='x', rotation=30)
axes[1].grid(alpha=0.3)

# Rating by price
price_analysis['Avg_Rating'].plot(kind='bar', ax=axes[2], color='teal',
                                   edgecolor='black', alpha=0.8)
axes[2].set_title('Rating by Price Range', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Average Rating', fontsize=11)
axes[2].tick_params(axis='x', rotation=30)
axes[2].axhline(df['rating'].mean(), color='red', linestyle='--', linewidth=2)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visuals/price_range_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visualization saved: price_range_analysis.png")
### 6️⃣ Cost vs Rating Correlation
# Scatter plot: Cost vs Rating
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(df['approx_cost(for two people)'], df['rating'],
                    c=df['votes'], s=df['votes']*0.5+20,
                    alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)

# Trend line
z = np.polyfit(df['approx_cost(for two people)'].dropna(), df['rating'].dropna(), 1)
p = np.poly1d(z)
x_line = np.linspace(df['approx_cost(for two people)'].min(), 
                     df['approx_cost(for two people)'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend Line')

ax.set_xlabel('Cost for Two People (₹)', fontsize=12, fontweight='bold')
ax.set_ylabel('Rating', fontsize=12, fontweight='bold')
ax.set_title('Cost vs Rating Correlation', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Votes', fontsize=11)

# Correlation
corr = df['approx_cost(for two people)'].corr(df['rating'])
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
       fontsize=12, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../visuals/cost_vs_rating.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visualization saved: cost_vs_rating.png")
print(f"📊 Correlation coefficient: {corr:.3f}")
### 7️⃣ Top Restaurants
# Top 10 by rating
top_rated = df.nlargest(10, 'rating')[['name', 'rating', 'votes', 
                                        'approx_cost(for two people)', 'listed_in(type)']]

print("🏆 Top 10 Highest Rated Restaurants:")
print(top_rated.to_string(index=False))
# Top 10 by votes
top_voted = df.nlargest(10, 'votes')[['name', 'rating', 'votes',
                                       'approx_cost(for two people)', 'listed_in(type)']]

print("👥 Top 10 Most Voted Restaurants:")
print(top_voted.to_string(index=False))
## 📝 Key Findings & Insights

### 🎯 Summary of Analysis

1. **Service Availability**
   - Online ordering is available in ~40% of restaurants
   - Only ~5% offer table booking services
   - Significant opportunity for digital transformation

2. **Quality Standards**
   - Average rating: ~3.6/5.0
   - Most restaurants maintain ratings above 3.5
   - Consistent quality standards across the platform

3. **Pricing Insights**
   - Average cost: ₹418 for two people
   - Mid-range restaurants receive highest engagement
   - Positive correlation between cost and ratings

4. **Restaurant Types**
   - Dining restaurants dominate the market
   - Different types show varying customer engagement
   - Category-specific trends in pricing and ratings

5. **Customer Engagement**
   - Total votes: 39,000+
   - Strong customer feedback culture
   - Higher ratings correlate with more votes
## 💡 Business Recommendations

1. **Digital Adoption**: Restaurants should invest in online ordering platforms to capture growing demand

2. **Table Booking**: Implementing reservation systems can enhance customer convenience

3. **Quality Focus**: Maintaining ratings above 4.0 significantly impacts customer choice

4. **Pricing Strategy**: Mid-range pricing (₹400-600) offers optimal balance of quality and affordability

5. **Customer Engagement**: Encouraging reviews and feedback can boost visibility and credibility
## ✅ Conclusion

This analysis of 148 Zomato restaurants provides valuable insights into the restaurant industry:

- Clear trends in service adoption and customer preferences
- Strong relationship between pricing, quality, and customer satisfaction
- Opportunities for restaurants to optimize their offerings
- Data-driven recommendations for business improvement

The comprehensive analysis reveals market dynamics that can inform strategic decisions for restaurant owners, investors, and platform operators.

---

**Project completed as part of GeeksforGeeks Data Analysis Certification Program**

**Author:** Durga Ahirwar  
**Date:** January 2026
