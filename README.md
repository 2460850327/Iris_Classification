# Iris Flower Classification 🌸

A complete machine learning workflow in Python to classify Iris flowers (**Setosa**, **Versicolor**, **Virginica**) based on their sepal and petal measurements, using `scikit-learn`.

## Project Objective

Build and compare multiple classification models to predict Iris species from flower measurements, covering the full ML pipeline:

1. Load and explore the dataset
2. Visualize the data
3. Preprocess (train/test split, scaling)
4. Train multiple classification models
5. Evaluate performance (accuracy, confusion matrix, classification report)
6. Understand which features matter most

## Dataset

The classic **Iris dataset** — 150 samples, 4 numeric features, 3 balanced classes (50 samples each):

| Column | Description |
|---|---|
| `SepalLengthCm` | Sepal length (cm) |
| `SepalWidthCm` | Sepal width (cm) |
| `PetalLengthCm` | Petal length (cm) |
| `PetalWidthCm` | Petal width (cm) |
| `Species` | Target label — Setosa, Versicolor, or Virginica |

Originally collected by Ronald Fisher (1936). Not included in this repo — download `Iris.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/iris) or load it directly via `sklearn.datasets.load_iris()`.

## Tech Stack

- **Python 3**
- **pandas / numpy** – data handling
- **matplotlib / seaborn** – visualization
- **scikit-learn** – preprocessing, modeling, and evaluation
- **Jupyter Notebook**

## Project Workflow

1. **Import Libraries** – Load pandas, numpy, matplotlib, seaborn, and scikit-learn modules
2. **Load the Dataset** – Read `Iris.csv`, drop the `Id` column
3. **Exploratory Data Analysis**
   - Pairplot of features by species
   - Correlation heatmap
   - Boxplots of each feature by species
   - Observation: petal measurements separate species far better than sepal measurements; Setosa is linearly separable
4. **Preprocessing**
   - Label-encode species names to integers
   - Stratified 80/20 train/test split
   - Standardize features with `StandardScaler`
5. **Train Multiple Models** – Compare five classifiers:
   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine (linear kernel)
   - Decision Tree
   - Random Forest
6. **Model Comparison** – Bar chart ranking models by test accuracy
7. **Detailed Evaluation** – Classification report and confusion matrix for the best-performing model
8. **Feature Importance** – Random Forest importances to identify which measurements matter most
9. **Predict on a New Sample** – Demonstrate inference on an unseen flower measurement

## Key Results

- All five models achieve strong accuracy on this well-separated, classic dataset.
- **Petal length and petal width** are consistently the most predictive features — far more so than sepal measurements.
- **Setosa** is almost always perfectly classified; **Versicolor** and **Virginica** are the two species most likely to be confused with each other.
- Since classes are balanced (50 samples each), accuracy is a reliable headline metric, but precision/recall/F1 give a fuller per-class picture.

## Key ML Concepts Demonstrated

- **Classification** vs. regression
- **Train/test split** for unbiased evaluation
- **Feature scaling** for distance- and gradient-based models
- **Accuracy, precision, recall, F1-score**
- **Confusion matrix** interpretation
- **Feature importance** via Random Forest

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Place `Iris.csv` in the project folder (or update the file path in the notebook).
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `Iris_Classification.ipynb` and run all cells.

## Project Structure

```
├── Iris_Classification.ipynb   # Main analysis & modeling notebook
├── README.md                    # Project documentation
└── data/                        # (Add Iris.csv here — not tracked in repo)
```

## Future Improvements

- Use `cross_val_score` for a more robust accuracy estimate instead of a single train/test split
- Tune hyperparameters with `GridSearchCV` (e.g., `n_neighbors` for KNN, `C`/`kernel` for SVM)
- Apply PCA to visualize the 4D feature space in 2D
- Deploy the best model as a simple web app (e.g., Streamlit or Flask)

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset: R.A. Fisher's classic Iris dataset, via [Kaggle](https://www.kaggle.com/datasets/uciml/iris) / `sklearn.datasets`
- Built with Python's data science and ML stack (pandas, seaborn, scikit-learn)
