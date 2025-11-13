# 🚗 Competitive Analysis of Entry-Level Luxury Car Market
### Unstructured Data Analytics Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![NLP](https://img.shields.io/badge/NLP-Text%20Mining-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Key Concepts Explained](#key-concepts-explained)
- [Methodology](#methodology)
- [Technical Implementation](#technical-implementation)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Learning Outcomes](#learning-outcomes)

---

## 🎯 Project Overview

This project analyzes **online forum discussions** from Edmunds.com to understand consumer perceptions and competitive dynamics in the entry-level luxury car market. By mining thousands of unstructured text messages, I extract strategic insights about brand positioning, consumer preferences, and aspirational brand value.

**Data Source:** 6,000+ forum messages discussing luxury car brands  
**Analysis Timeframe:** Complete competitive landscape analysis  
**Focus Brands:** BMW, Audi, Mercedes, Acura, Infiniti, Lexus, Cadillac, and more

---

## 💼 Business Problem

An automotive manufacturer wants to enter the competitive entry-level luxury segment. They need answers to:

- 🤔 **Which brands are consumers discussing together?**
- 💭 **What attributes do consumers associate with each brand?**
- ⭐ **Which brand has the strongest aspirational appeal?**
- 🎯 **Where are the opportunities for differentiation?**

Traditional surveys are expensive and slow. This project demonstrates how **unstructured data mining** can provide these insights faster and cheaper by analyzing real consumer conversations.

---

## 🧠 Key Concepts Explained (In Simple Language)

### 1. **Web Scraping** 🕷️
**What it is:** Automatically extracting data from websites (like copying information from thousands of web pages in seconds)

**What I did:** 
- Built a scraper to collect forum posts from Edmunds.com
- Handled nested HTML structures (messages within messages)
- Prevented double-counting quoted text
- Extracted clean message content for analysis

**Real-world application:** Companies scrape competitor websites, job listings, product reviews, and social media to gather market intelligence.

---

### 2. **Zipf's Law** 📊
**What it is:** A natural pattern in language where the most common word appears twice as often as the second most common, three times as often as the third most common, and so on.

**In simple terms:** If "car" appears 1000 times, "drive" might appear ~500 times, and "engine" ~333 times. It's like a universal rule for word frequency!

**What I did:**
- Extracted all words from 6,000+ messages
- Counted frequency of each word
- Tested if our data follows Zipf's Law using statistical regression
- **Result:** Our forum data DOES follow Zipf's Law (β = -1.06, statistically significant)

**Why it matters:** Confirms our dataset represents natural language, not spam or artificially generated content.

---

### 3. **Lift Ratio** 🔗
**What it is:** A measure of how strongly two things appear together compared to random chance.

**The formula:** `Lift = P(A and B) / [P(A) × P(B)]`

**In simple terms:**
- **Lift = 1:** Brands mentioned together by pure coincidence (no relationship)
- **Lift > 1:** Brands mentioned together MORE than expected (strong association)
- **Lift < 1:** Brands mentioned together LESS than expected (competitors, not compared)

**Example from my analysis:**
- **Mercedes & Cadillac: Lift = 5.68** → People talk about these together ALL THE TIME (luxury comparison)
- **BMW & Toyota: Lift = 0.82** → Rarely compared (different market segments)

**What I did:**
- Built a co-occurrence matrix counting brand pairs in messages
- Calculated lift ratios for all brand combinations
- Identified which brands consumers mentally group together

**Business insight:** High lift = consumers see brands as substitutes (direct competitors). Low lift = different market positions.

---

### 4. **Multidimensional Scaling (MDS)** 🗺️
**What it is:** Taking complex relationships and creating a simple 2D map where similar things are close together.

**In simple terms:** Imagine you have distances between 50 cities, but no map. MDS creates that map! In our case, it maps brands based on how often they're discussed together.

**What I did:**
- Converted lift ratios into "dissimilarity scores" (1/lift)
- Used MDS algorithm to position brands on a 2D plot
- **Clusters found:**
  - **German Luxury:** BMW, Audi, Mercedes (close together)
  - **Japanese Mainstream:** Honda, Toyota, Nissan
  - **American Brands:** Cadillac, Ford (separate cluster)

**Visual output:** A perceptual map showing competitive positioning in consumers' minds.

**Business value:** Shows which brands compete directly and where white space exists for new entrants.

---

### 5. **Stopwords Removal** 🛑
**What it is:** Filtering out common words that don't add meaning (like "the," "is," "and," "a")

**Why we do it:** These words appear everywhere but tell us nothing about car brands or attributes.

**What I did:**
- Created a custom stopword list including automotive forum jargon
- Removed stopwords before analyzing brand and attribute frequencies
- Kept meaningful words like "luxury," "performance," "reliability"

**Impact:** Cleaner, more meaningful analysis focusing on important content.

---

### 6. **Tokenization** ✂️
**What it is:** Breaking text into individual words (tokens) for analysis.

**Example:**
- **Original:** "I love my BMW 3-series!"
- **Tokenized:** ["I", "love", "my", "BMW", "3-series"]

**What I did:**
- Split messages into individual words
- Handled contractions (e.g., "don't" → "do not")
- Preserved brand names and model numbers
- Converted to lowercase for consistency

**Why it matters:** You can't analyze text until it's broken into analyzable pieces!

---

### 7. **Regular Expressions (Regex)** 🔍
**What it is:** A powerful pattern-matching language for finding specific text patterns.

**In simple terms:** Like "Find and Replace" on steroids—you can find anything matching a pattern.

**What I did:**
- **Extract words:** `\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b` (finds all words, including contractions)
- **Model matching:** Built patterns to find multi-word car models (e.g., "3 series", "a4 quattro")
- **Brand extraction:** Replaced car models with brand names using regex substitution

**Real-world use:** Email validation, phone number extraction, data cleaning—regex is everywhere!

---

### 8. **Co-occurrence Analysis** 👥
**What it is:** Counting how often two things appear together in the same context.

**In simple terms:** If someone mentions "BMW" and "Audi" in the same message, that's a co-occurrence.

**What I did:**
- Created a matrix counting brand pairs in messages
- **Key rule:** Each brand counted only ONCE per message (prevents spammy counting)
- Built symmetric matrix (BMW-Audi = Audi-BMW)

**Example results:**
- BMW mentioned 2,686 times alone
- BMW & Audi mentioned together 308 times
- This high co-occurrence suggests direct competition

---

### 9. **Proximity Window Analysis** 📏
**What it is:** Looking at words that appear NEAR each other (within X words), not just anywhere in the document.

**Why it matters:** "The BMW has great luxury" is different from "I prefer luxury. Also, BMW is fast." The first associates luxury WITH BMW; the second doesn't.

**What I did:**
- Set window size = 15 words
- Only counted brand-attribute pairs within 15 words of each other
- Calculated lift for these close associations

**Example:**
- **"The BMW has great performance"** → BMW + performance counted (within window)
- **"I like performance. BMW is okay"** → NOT counted (too far apart)

**Business insight:** Identifies TRUE associations vs. coincidental mentions.

---

### 10. **Aspirational Brand Analysis** ⭐
**What it is:** Identifying which brands people WANT to own (dream about) vs. just discuss.

**Two methods I used:**

**Method 1: Pre-defined phrases**
- Looked for patterns like "I want a [brand]", "dream of owning", "hope to buy"
- Calculated lift ratios for aspirational mentions
- **Winner:** Cadillac (2.61 lift)

**Method 2: Data-driven N-grams**
- Extracted actual phrases from data containing aspirational words
- Built patterns from real consumer language
- Validated Method 1 results
- **Winner:** Cadillac again (2.84 lift)

**Business value:** Shows which brand has the strongest emotional pull and aspiration.

---

### 11. **N-grams** 📝
**What it is:** Sequences of N consecutive words (bigrams = 2 words, trigrams = 3 words)

**Examples:**
- **Unigram:** "BMW"
- **Bigram:** "luxury car"
- **Trigram:** "ultimate driving machine"

**What I did:**
- Extracted bigrams and trigrams containing aspirational words
- Found phrases like "want to buy", "hope to own", "dream car"
- Used these to identify aspirational brand mentions

**Why useful:** Captures context better than single words. "Not good" means something very different from "good"!

---

### 12. **Statistical Hypothesis Testing** 📈
**What it is:** Using math to determine if a pattern is real or just random chance.

**For Zipf's Law, I tested:**
- **Null hypothesis:** β = -1 (perfect Zipf's Law)
- **My result:** β = -1.06
- **t-statistic:** -3.96 (significant at p < 0.001)

**In simple terms:** There's less than 0.1% chance this result happened randomly. The pattern is REAL!

**What I learned:** How to validate findings with statistical rigor, not just eyeballing graphs.

---

### 13. **Data Cleaning & Preprocessing** 🧹
**What it is:** Transforming messy real-world data into clean, analyzable format.

**Challenges I handled:**
- **Multi-brand entries:** "hyundai kia" → split into separate rows
- **Special characters:** Removed punctuation and symbols
- **Case sensitivity:** Converted all to lowercase
- **Duplicates:** Removed duplicate brand-model pairs
- **Missing values:** Filtered out null entries

**Real-world lesson:** Data cleaning takes 80% of the time in any analytics project!

---

## 🔬 Methodology

### Phase 1: Data Collection & Preparation
1. **Web Scraping**
   - Scraped Edmunds.com luxury car forums
   - Extracted 6,000+ messages with metadata
   - Handled quoted text to prevent double-counting

2. **Data Cleaning**
   - Tokenized text (split into words)
   - Removed stopwords and special characters
   - Normalized brand and model names

---

### Phase 2: Exploratory Analysis

3. **Zipf's Law Validation**
   - Word frequency analysis on entire corpus
   - Log-log regression to test power law
   - Statistical hypothesis testing (t-test)
   - **Result:** Confirmed natural language distribution

4. **Brand Frequency Analysis**
   - Mapped car models to parent brands using lookup table
   - Counted brand mentions (unique per message)
   - Identified top 10 most discussed brands

**Top 10 Brands:**
1. BMW (2,686 mentions)
2. Audi (1,536)
3. Acura (1,346)
4. Honda (992)
5. Infiniti (855)
6. Volkswagen (794)
7. Toyota (753)
8. Mercedes (561)
9. Cadillac (363)
10. Ford (344)

---

### Phase 3: Association & Positioning Analysis

5. **Lift Ratio Calculation**
   - Built brand co-occurrence matrix
   - Calculated lift for all brand pairs
   - Identified strongest associations

**Key Findings:**
- **Mercedes-Cadillac (5.68):** Luxury comparison
- **Audi-Mercedes (3.51):** German luxury rivalry
- **Honda-Acura (3.27):** Parent-subsidiary link

6. **Multidimensional Scaling (MDS)**
   - Converted lift to dissimilarity (1/lift)
   - Applied MDS algorithm (2D visualization)
   - Created perceptual map of brand positioning

**Clusters Discovered:**
- **Premium German:** BMW, Audi, Mercedes
- **Japanese Value-Luxury:** Acura, Infiniti, Lexus
- **American Heritage:** Cadillac, Ford

---

### Phase 4: Attribute Association Analysis

7. **Attribute Extraction**
   - Identified top 5 attributes: luxury, performance, quality, style, reliability
   - Used proximity window (15 words) for context
   - Calculated lift for brand-attribute pairs

**Strongest Associations:**
- **Cadillac-Luxury (2.25)**
- **Audi-Premium (1.76)**
- **BMW-Performance (1.64)**

8. **Aspirational Brand Analysis**
   - Method 1: Pre-defined aspirational phrases
   - Method 2: Data-driven n-gram extraction
   - Cross-validation of both approaches

**Winner:** Cadillac (highest aspirational lift in both methods)

---

## 💻 Technical Implementation

### Core Technologies & Libraries

```python
# Data Manipulation
import pandas as pd
import numpy as np

# Web Scraping
import requests
from bs4 import BeautifulSoup

# Text Processing
import re
from collections import Counter

# Statistical Analysis
import statsmodels.api as sm
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Natural Language Processing
import nltk
from nltk.corpus import stopwords
```

---

### Key Code Snippets

#### 1. Web Scraping Function
```python
def extract_message_without_double_counting(post):
    """
    Extracts forum message while preventing quoted text duplication.
    Returns clean message content.
    """
    message_div = post.find('div', class_='Message userContent')
    # Remove quoted text and blockquotes
    # Extract only original content
    return cleaned_message
```

#### 2. Lift Ratio Calculation
```python
def calculate_lift(brand_a, brand_b, total_messages):
    """
    Calculates lift ratio for brand co-occurrence.
    Lift = P(A ∩ B) / [P(A) × P(B)]
    """
    p_a = count(brand_a) / total_messages
    p_b = count(brand_b) / total_messages
    p_ab = count(brand_a AND brand_b) / total_messages
    
    lift = p_ab / (p_a * p_b)
    return lift
```

#### 3. Proximity Window Analysis
```python
def attribute_brand_lift_windowed(df, attributes, window_size=15):
    """
    Calculates brand-attribute associations within proximity window.
    Only counts mentions within 15 words of each other.
    """
    for tokens in df['tokens']:
        for i, token in enumerate(tokens):
            if token in attributes:
                window = tokens[max(0, i-window_size):i+window_size+1]
                # Extract brands within window
                # Calculate lift
```

#### 4. MDS Visualization
```python
from sklearn.manifold import MDS

# Convert lift to dissimilarity
dissimilarity_matrix = 1 / lift_matrix

# Apply MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
brand_positions = mds.fit_transform(dissimilarity_matrix)

# Create perceptual map
plt.scatter(brand_positions[:, 0], brand_positions[:, 1])
for i, brand in enumerate(brands):
    plt.annotate(brand, (brand_positions[i, 0], brand_positions[i, 1]))
```

---

## 📊 Key Findings

### 1. Competitive Landscape
- **German brands cluster together** → Seen as direct substitutes
- **Japanese brands occupy value-luxury space** → Different positioning
- **American brands isolated** → Opportunity for differentiation

### 2. Brand-Attribute Associations
- **Cadillac = Luxury** (strongest association)
- **BMW = Performance** (aligned with brand promise)
- **Audi = Premium** (positioned between luxury and performance)

### 3. Aspirational Leader
- **Cadillac wins** both aspirational analysis methods
- Despite lower sales, highest emotional appeal in discussions
- Opportunity: Convert aspiration into purchase intent

### 4. Market Entry Strategy
**Recommendations:**
- Target the "accessible luxury" segment
- Differentiate on technology + value proposition
- Avoid head-on competition with German incumbents
- Leverage American heritage with modern execution (Cadillac model)

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **BeautifulSoup** | HTML parsing for web scraping |
| **NLTK** | Natural language processing |
| **Scikit-learn** | Machine learning (MDS algorithm) |
| **Statsmodels** | Statistical testing and regression |
| **Matplotlib** | Data visualization |
| **Regular Expressions** | Pattern matching and text extraction |

---

## 📁 Project Structure

```
luxury-car-market-analysis/
│
├── FinalSubmission_HW1_UD.ipynb    # Main analysis notebook
├── sample_data.csv                  # Forum messages dataset (6,000+ rows)
├── car_models_and_brands.csv       # Brand-model mapping (500+ entries)
│
├── outputs/
│   ├── zipf_law_plot.png           # Zipf's Law visualization
│   ├── mds_brand_map.png           # 2D perceptual map
│   ├── lift_heatmap.png            # Brand association matrix
│   └── attribute_analysis.png      # Brand-attribute associations
│
└── README.md                        # This file
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn nltk beautifulsoup4 requests statsmodels scipy
```

### Running the Analysis
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/luxury-car-market-analysis.git
cd luxury-car-market-analysis

# 2. Launch Jupyter Notebook
jupyter notebook

# 3. Open FinalSubmission_HW1_UD.ipynb

# 4. Run all cells sequentially (Cell → Run All)
```

### Expected Runtime
- Data loading: < 1 minute
- Text processing: 2-3 minutes
- Statistical analysis: < 1 minute
- MDS computation: < 30 seconds
- **Total runtime: ~5 minutes**

---

## 📚 Learning Outcomes

Through this project, I gained hands-on experience with:

### Technical Skills
✅ **Web scraping** complex HTML structures  
✅ **Text preprocessing** (tokenization, stopword removal, regex)  
✅ **Statistical hypothesis testing** (t-tests, regression)  
✅ **Association rule mining** (lift, support, confidence)  
✅ **Dimensionality reduction** (MDS algorithm)  
✅ **Data visualization** (perceptual maps, heatmaps)  

### Business Skills
✅ **Competitive intelligence** from unstructured data  
✅ **Brand positioning** analysis  
✅ **Consumer perception** mapping  
✅ **Strategic recommendations** from data insights  

### Key Takeaways
1. **Unstructured data is gold** → Real consumer language reveals truths surveys miss
2. **Simple metrics are powerful** → Lift ratio explains complex relationships simply
3. **Validation matters** → Multiple methods (2 aspiration analyses) build confidence
4. **Visualization clarifies** → MDS map instantly shows competitive dynamics
5. **Context is everything** → Proximity windows separate real associations from noise

---

## 📈 Future Enhancements

**Potential next steps:**
- [ ] **Sentiment analysis** → Which brands generate positive/negative sentiment?
- [ ] **Topic modeling (LDA)** → What themes dominate discussions for each brand?
- [ ] **Time series analysis** → How do perceptions change over time?
- [ ] **Word embeddings (Word2Vec)** → Find semantic relationships between brands
- [ ] **Predictive modeling** → Forecast brand preference based on attributes
- [ ] **Expand data sources** → Include Reddit, Twitter, YouTube comments

---

## 🤝 Contributing

This project is part of my academic portfolio, but I welcome feedback and suggestions!

- 🐛 Found an issue? Open an issue
- 💡 Have an idea? Start a discussion
- 🔧 Want to improve? Submit a pull request

---

## 📄 License

This project is available for educational and portfolio purposes. Data sourced from public forums (Edmunds.com).

---

## 👤 Author

**[Your Name]**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🌐 Portfolio: [yourportfolio.com](https://yourportfolio.com)

---

## 🙏 Acknowledgments

- **Data Source:** Edmunds.com automotive forums
- **Course:** Analytics for Unstructured Data (MSBA F2025)
- **Team Members:** Christian Breton, Mohar Chaudhuri, Stiles Clements, Muskan Khepar

---

## 📝 Citation

If you find this project helpful, please cite:

```bibtex
@misc{luxury_car_analysis_2025,
  author = {Your Name},
  title = {Competitive Analysis of Entry-Level Luxury Car Market Using Unstructured Data},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/luxury-car-market-analysis}
}
```

---

**⭐ If you found this project interesting, please consider giving it a star!**

---

*Last Updated: November 2025*
