# Netflix Recommendation System

A comprehensive machine learning project implementing collaborative filtering models to build intelligent movie recommendation systems. This project explores both traditional matrix factorization and scalable distributed approaches for large-scale recommendation tasks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Models Implemented](#models-implemented)
- [Results & Performance](#results--performance)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)

---

## Overview

This project develops a Netflix recommendation system using real movie rating data extracted from a PostgreSQL data warehouse. The system implements two complementary approaches:

1. **Non-Spark Solution**: Scikit-Learn TruncatedSVD for local matrix factorization
2. **Spark MLlib Solution**: Distributed ALS (Alternating Least Squares) for scalable recommendations

The project also includes graph analytics to identify movie communities, hub movies, and co-viewing patterns.

---

## Features

✅ **Data Ingestion & Management**

- Connect to PostgreSQL data warehouse via SQLAlchemy
- Extract clean samples with readable movie titles
- Automatic data quality validation

✅ **Feature Engineering**

- User average rating bias correction
- Movie average rating computation
- Normalized rating features for model stability

✅ **Model Training & Tuning**

- Hyperparameter optimization with grid search
- TrainValidationSplit for robust model selection
- Train/test split with proper data leakage prevention

✅ **Performance Metrics**

- Multi-metric evaluation: MAE, MAD, MAPE, MSE, RMSE
- Comparative analysis of both models
- Visual performance comparison charts

✅ **Graph Analytics**

- Bipartite user-movie network construction
- Movie centrality analysis (degree centrality)
- Movie-to-movie graph projection
- Community detection with greedy modularity
- Network visualization with community coloring

---

## Dataset

**Source**: Netflix Prize Dataset (100,000 sample ratings)

- **Total Customers**: 480,000+ (sample: subset used in modeling)
- **Total Movies**: 17,770+ (sample: top movies analyzed)
- **Rating Scale**: 1-5 stars (integral values)
- **Time Period**: October 1998 - December 2005

**Data Schema**:

- `fact_ratings`: Rating facts (rating_key, customer_key, movie_key, rating, date_key)
- `dim_customer`: Customer dimension (customer_key, customer_id)
- `dim_movie`: Movie dimension (movie_key, movie_id, title, year)

**Data Quality**:

- Missing/invalid titles filtered automatically
- 100,000 ratings extracted for analysis
- No null values in key columns
- Rating distribution heavily skewed toward 3-4 stars (users rate movies they liked)

---

## Project Structure

```
netflix/modelling/
├── model.ipynb                 # Main Jupyter notebook with all analyses
├── README.md                   # This file
├── requirements.txt            # Python package dependencies
├── about_dataset.txt          # Dataset documentation
├── schemas.txt                # Database schema definitions
└── assignment.txt             # Project assignment details
```

**Notebook Sections**:

1. **Data Ingestion & Sampling** - Database connection and data extraction
2. **Data Quality Checks** - Exploratory data analysis
3. **Feature Engineering** - Bias correction and normalized features
4. **Machine Learning Pipeline**
   - Part A: Non-Spark Baseline (TruncatedSVD)
   - Part B: Spark MLlib (ALS)
5. **Evaluation & Comparison** - Metrics and performance visualization
6. **Graph Analytics** - Network analysis and community detection

---

## Requirements

### Python Version

- Python 3.7 - 3.11 (PySpark compatibility requirement)

### Dependencies

```
pandas            # Data manipulation and analysis
numpy             # Numerical computing
SQLAlchemy        # SQL toolkit and ORM
psycopg2-binary   # PostgreSQL adapter
scikit-learn      # Machine learning models
pyspark           # Distributed computing (requires Java)
ipykernel         # Jupyter kernel support
networkx          # Network and graph analysis
matplotlib        # Data visualization
```

### System Requirements

- 4GB+ RAM (8GB+ recommended for Spark)
- PostgreSQL database connection
- Java Runtime Environment (JRE) for Spark

---

## Installation

### Step 1: Clone Repository

```bash
cd /path/to/Bu\ Lina/netflix/modelling
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n netflix-rec python=3.10
conda activate netflix-rec
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, sklearn, pyspark, networkx; print('All packages installed successfully!')"
```

---

## Setup Instructions

### Database Configuration

The notebook connects to PostgreSQL using environment variables. Create a `.env` file in the project directory:

```env
NETFLIX_DB_HOST=your_db_host
NETFLIX_DB_PORT=5432
NETFLIX_DB_NAME=netflix
NETFLIX_DB_USER=postgres
NETFLIX_DB_PASSWORD=your_password
```

### Alternative: Direct Configuration

Edit the connection parameters in cell 3 of the notebook:

```python
DB_HOST = "your_host"
DB_PORT = "5432"
DB_NAME = "netflix"
DB_USER = "postgres"
DB_PASSWORD = "your_password"
```

### Environment Setup for Spark

On Windows systems, ensure proper Python interpreter configuration:

```bash
set PYSPARK_PYTHON=path\to\python.exe
set PYSPARK_DRIVER_PYTHON=path\to\python.exe
```

---

## Usage

### Run the Complete Analysis

1. **Start Jupyter Notebook**

   ```bash
   jupyter notebook model.ipynb
   ```

2. **Execute Cells Sequentially**
   - Cells are organized in logical sections
   - Markdown cells explain each step
   - Run from top to bottom

3. **Key Execution Checkpoints**
   - **Cell 3**: Data extraction ("Sample extracted: 100000")
   - **Cell 5**: Data quality check (shows data types and completeness)
   - **Cell 8**: Feature engineering preview
   - **Cell 11**: Scikit-Learn hyperparameter tuning
   - **Cell 13**: Best model metrics output
   - **Cell 16**: Spark ALS training (may take 2-5 minutes)
   - **Cell 19**: Metrics comparison
   - **Cell 20**: Performance visualization
   - **Cell 22+**: Graph analytics and visualization

### Run Individual Sections

- **Data Analysis Only**: Run cells 1-6
- **Scikit-Learn Model**: Run cells 1-13
- **Spark Model**: Run cells 1-17
- **Graph Analytics**: Run cells 22-27

---

## Project Workflow

```
1. DATA INGESTION
   ├─ Connect to PostgreSQL
   ├─ Query Star Schema (fact_ratings, dim_movie, dim_customer)
   └─ Extract 100K ratings with clean movie titles

2. DATA QUALITY CHECKS
   ├─ Validate data types and completeness
   ├─ Analyze rating distribution
   └─ Identify data quality issues

3. FEATURE ENGINEERING
   ├─ Calculate user_avg_rating (user bias correction)
   ├─ Calculate movie_avg_rating (movie bias correction)
   └─ Compute normalized_rating = rating - user_avg_rating

4. TRAIN/TEST SPLIT
   ├─ 80/20 random split with stratification
   ├─ Filter test to seen users/items
   └─ Create user-item rating matrix

5. MODEL A: SCIKIT-LEARN TSVD
   ├─ Hyperparameter tuning (n_components: 50-150)
   ├─ Train TruncatedSVD on user-item matrix
   ├─ Predict ratings for test set
   └─ Calculate metrics (MAE, MAD, MAPE, MSE, RMSE)

6. MODEL B: SPARK ALS
   ├─ Create Spark DataFrame
   ├─ Hyperparameter tuning (rank, regParam)
   ├─ Train ALS model with validation
   ├─ Transform predictions back to rating scale
   └─ Calculate metrics

7. EVALUATION & COMPARISON
   ├─ Compare metrics across models
   ├─ Create bar charts
   └─ Determine best approach

8. GRAPH ANALYTICS
   ├─ Build bipartite user-movie graph
   ├─ Calculate movie centrality
   ├─ Project to movie-movie graph
   ├─ Detect communities
   └─ Visualize with community colors
```

---

## Models Implemented

### Model A: Non-Spark TruncatedSVD

**Algorithm**: Truncated Singular Value Decomposition

- **Library**: Scikit-Learn
- **Approach**: Matrix factorization with limited rank

**Process**:

1. Create user-item matrix with normalized ratings
2. Test multiple n_components values (50, 100, 150)
3. Decompose matrix: `U × Σ × V^T ≈ R`
4. Predict: `R_pred = U × Σ × V^T + User_Bias`
5. Select model with lowest test RMSE

**Advantages**:

- Fast on local machines
- Good interpretability
- Low memory footprint

**Limitations**:

- Doesn't scale to very large datasets
- No explicit regularization
- Cold-start problem for new users

---

### Model B: Spark MLlib ALS

**Algorithm**: Alternating Least Squares

- **Library**: PySpark MLlib
- **Approach**: Distributed matrix factorization

**Process**:

1. Create Spark DataFrame from ratings
2. Split train/validation within Spark
3. Grid search over:
   - `rank`: [30, 50] (number of latent factors)
   - `regParam`: [0.05, 0.1] (regularization strength)
4. Train multiple ALS models in parallel
5. Evaluate on validation set
6. Select best model and test

**Advantages**:

- Scales to billions of ratings
- Built-in regularization
- Distributed training across clusters
- Better generalization

**Limitations**:

- Requires Java and cluster setup
- Slower on local machines
- More hyperparameters to tune

---

## Results & Performance

### Performance Metrics Comparison

Both models are evaluated on the same test set using 5 metrics:

| Metric   | Scikit-Learn SVD | Spark ALS | Lower is Better |
| -------- | ---------------- | --------- | --------------- |
| **MAE**  | ~0.75            | ~0.72     | ✓               |
| **MAD**  | ~0.60            | ~0.58     | ✓               |
| **MAPE** | ~22.5%           | ~21.0%    | ✓               |
| **MSE**  | ~0.82            | ~0.80     | ✓               |
| **RMSE** | ~0.90            | ~0.89     | ✓               |

**Interpretation**:

- RMSE ~0.9 means predictions are typically off by ~0.9 stars
- Reasonable performance for a 5-star scale
- Both models show similar accuracy, confirming Spark can scale without quality loss

### Hyperparameter Results

**Scikit-Learn Best Config**:

- n_components: 100 (optimal balance)
- Training time: <5 minutes
- Memory: ~500MB

**Spark Best Config**:

- rank: 50 (latent factors)
- regParam: 0.05 (regularization)
- Training time: 2-5 minutes (depends on system)
- Can scale to 1M+ ratings

---

## Key Findings

### 1. Rating Distribution

- Users predominantly rate 3-5 stars
- Positive bias: users rate movies they liked
- 1-2 star ratings are rare (clear strong dislikes)
- Implication: Model trained on positive bias

### 2. Hub Movies

Analysis of movie centrality revealed popular "hub" movies:

- **"North by Northwest"** (highest centrality)
- Top 10 movies watched by 60%+ of sample users
- These hub movies are excellent for new user recommendations

### 3. Movie Communities

Graph analysis detected **2 major communities**:

**Community 0** (17 movies):

- High-quality classic films
- Strong co-viewing patterns
- Likely connected by genre/theme

**Community 1** (13 movies):

- Recent/popular titles
- Different audience demographics
- Distinct from Community 0

**Implication**: Movies naturally cluster into audiences - good for targeted "similar movies" recommendations

### 4. Co-viewing Patterns

- Movies frequently co-watched indicate shared audience
- Network density suggests strong thematic connections
- Bridge movies connect different communities

### 5. User Behavior

- Top 10 users account for ~30% of sample ratings
- Most users rate 5-20 movies (sparse feedback)
- User bias varies significantly (some rate everything high)
- Normalization crucial for fair comparison

---

## Technologies Used

### Data Management

- **PostgreSQL**: Data warehouse with Star Schema
- **SQLAlchemy**: Python SQL toolkit
- **Pandas**: Data manipulation and analysis

### Machine Learning

- **Scikit-Learn**: Traditional ML models (TruncatedSVD)
- **PySpark MLlib**: Distributed ML (ALS)
- **NumPy**: Numerical operations

### Analytics & Visualization

- **NetworkX**: Graph construction and analysis
- **Matplotlib**: 2D plotting and visualization
- **Jupyter**: Interactive notebook environment

### Development

- **Python 3.7-3.11**: Programming language
- **Virtual Environment**: Dependency isolation

---

## Troubleshooting

### Issue: Database Connection Failed

```
Solution: Verify .env file or DB_* variables in cell 3
- Check database is running
- Verify credentials
- Ensure tunnel is active
```

### Issue: PySpark Not Found

```
Solution: Install Java and set JAVA_HOME
sudo apt-get install default-jre  # Linux
brew install java                 # macOS
```

### Issue: Out of Memory During Spark

```
Solution: Reduce sample size in Spark cell
Change spark_sample_fraction from 0.5 to 0.25
```

### Issue: Python Version Not Compatible

```
Solution: Use Python 3.10 or earlier
conda create -n netflix-rec python=3.10
```

---

## Future Enhancements

- [ ] Deep Learning models (Neural Collaborative Filtering)
- [ ] Content-based filtering integration
- [ ] Real-time recommendation serving
- [ ] A/B testing framework
- [ ] Cold-start problem solutions
- [ ] Implicit feedback modeling
- [ ] Cross-domain recommendation
- [ ] Explainable AI features

---

## References

- Netflix Prize Dataset: http://www.netflixprize.com
- Scikit-Learn TruncatedSVD: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
- PySpark MLlib ALS: https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
- NetworkX Graph Analysis: https://networkx.org/
