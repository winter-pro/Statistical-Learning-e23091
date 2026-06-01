import pandas as pd
import numpy as np
import io
from google.colab import files
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from IPython.display import display, HTML

class PlottingMethods:
    """
    A modular class for generating specific interactive Plotly charts.
    """
    
    def display_image(self, result_dict):
        """Renders the HTML string returned by plotting methods."""
        if result_dict.get('status') == 'success':
            display(HTML(result_dict['html']))
        else:
            print("Error generating plot.")

    def plot_bar_chart(self, data, x, y, color=None, barmode='group'):
        """Generates a bar chart."""
        if data is None or data.empty: return {'status': 'error'}
        fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, title=f"Bar Chart: {y} by {x}")
        return {'status': 'success', 'html': fig.to_html(full_html=False)}

    def plot_pie_chart(self, data, names, values, hole=0.0):
        """Generates a pie or donut chart."""
        if data is None or data.empty: return {'status': 'error'}
        fig = px.pie(data, names=names, values=values, hole=hole, title=f"Pie Chart: {values} distribution")
        return {'status': 'success', 'html': fig.to_html(full_html=False)}

    def plot_histogram(self, data, x, bins=None):
        """Generates a histogram."""
        if data is None or data.empty: return {'status': 'error'}
        fig = px.histogram(data, x=x, nbins=bins, title=f"Histogram: {x} Distribution")
        return {'status': 'success', 'html': fig.to_html(full_html=False)}


class DataInspector:
    """
    An end-to-end tool for CSV data ingestion, advanced cleaning, 
    feature engineering preparation, and high-level statistical visualization.
    """
    def __init__(self):
        self.df = pd.DataFrame()

    # ==========================================
    # 1. Data Ingestion & Sanitization
    # ==========================================
    def upload_data(self):
        """Handles local file uploads in Google Colab and sanitizes garbage strings."""
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return
            
        filename = list(uploaded.keys())[0]
        garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
        self.df = pd.read_csv(io.BytesIO(uploaded[filename]), na_values=garbage_strings)
        self._auto_correct_types()
        print(f"Successfully loaded {filename}")

    def _auto_correct_types(self):
        """Force-converts columns to numeric types if not entirely null."""
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if not converted.isna().all():
                self.df[col] = converted

    # ==========================================
    # 2. Structural Analysis & Cleaning
    # ==========================================
    def get_summary(self):
        """Displays row/col counts, preview, and data type breakdown."""
        if self.df.empty: return "Dataframe is empty."
        print(f"Dataset Shape: {self.df.shape[0]} Rows, {self.df.shape[1]} Columns\n")
        
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}\n")
        print("First 20 Rows Preview:")
        display(self.df.head(20))

    def handle_missing_values(self, strategy='median', constant_value=0):
        """Imputes missing values based on mean, median, mode, or constant."""
        if self.df.empty: return
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0: continue
            
            if self.df[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'constant':
                    self.df[col] = self.df[col].fillna(constant_value)
            else:
                if strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == 'constant':
                    self.df[col] = self.df[col].fillna(str(constant_value))
        print(f"Missing values handled using '{strategy}' strategy.")

    def remove_duplicates(self):
        """Prunes exact row matches."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_rows - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns=None, find_and_delete=False):
        """IQR-based outlier detection system."""
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()
            
        initial_rows = len(self.df)
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if find_and_delete:
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            else:
                self.df[f'{col}_is_outlier'] = ~((self.df[col] >= lower_bound) & (self.df[col] <= upper_bound))
                
        if find_and_delete:
            print(f"Removed {initial_rows - len(self.df)} outlier rows.")
        else:
            print("Outliers flagged in new columns.")

    # ==========================================
    # 3. Feature Engineering Preparation
    # ==========================================
    def extract_normalized_numeric_data(self, method='standard'):
        """Scales numeric data using minmax, standard, or robust scaling."""
        num_cols = self.df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0: return pd.DataFrame()
        
        scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        scaler = scalers.get(method, StandardScaler())
        scaled_data = scaler.fit_transform(self.df[num_cols])
        return pd.DataFrame(scaled_data, columns=num_cols, index=self.df.index)

    def extract_normalized_categorical_data(self, method='onehot'):
        """Encodes categorical data."""
        cat_cols = self.df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) == 0: return pd.DataFrame()
        
        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = encoder.fit_transform(self.df[cat_cols])
            col_names = encoder.get_feature_names_out(cat_cols)
            return pd.DataFrame(encoded_data, columns=col_names, index=self.df.index)
        elif method == 'ordinal':
            encoder = OrdinalEncoder()
            encoded_data = encoder.fit_transform(self.df[cat_cols])
            return pd.DataFrame(encoded_data, columns=cat_cols, index=self.df.index)

    def create_normalized_data_df(self, num_method='standard', cat_method='onehot'):
        """Merges scaled numeric and encoded categorical data."""
        num_df = self.extract_normalized_numeric_data(method=num_method)
        cat_df = self.extract_normalized_categorical_data(method=cat_method)
        return pd.concat([num_df, cat_df], axis=1)

    # ==========================================
    # 4. Advanced Interactive Visualization
    # ==========================================
    def plot_numerical(self, columns):
        """Generates a 3-panel subplot (Violin, Scatter, Hist) for numeric columns."""
        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Violin Plot", "Scatter (Index vs Value)", "Histogram"))
            fig.add_trace(go.Violin(x=self.df[col], name=col, orientation='h'), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.df[col], mode='markers', name=col), row=1, col=2)
            fig.add_trace(go.Histogram(x=self.df[col], name=col), row=1, col=3)
            fig.update_layout(title_text=f"Distribution Analysis: {col}", height=400, showlegend=False)
            fig.show()

    def plot_relationship(self, col1, col2):
        """Smart relationship detector: Num-Num, Cat-Num, or Cat-Cat."""
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
        
        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Scatter: {col1} vs {col2}")
        elif not is_num1 and is_num2:
            fig = px.box(self.df, x=col1, y=col2, points="all", title=f"Boxplot: {col2} grouped by {col1}")
        elif is_num1 and not is_num2:
            fig = px.box(self.df, x=col2, y=col1, points="all", title=f"Boxplot: {col1} grouped by {col2}")
        else:
            counts = self.df.groupby([col1, col2]).size().reset_index(name='count')
            fig = px.bar(counts, x=col1, y='count', color=col2, barmode='group', title=f"Grouped Bar: {col1} vs {col2}")
        
        fig.show()

    # ==========================================
    # 5. Deep Statistical Insights
    # ==========================================
    def plot_all_associations_heatmap(self):
        """Visualizes Pearson's r correlations for numeric data."""
        num_cols = self.df.select_dtypes(include=np.number).columns
        if len(num_cols) < 2:
            print("Not enough numeric columns for a heatmap.")
            return
            
        corr_matrix = self.df[num_cols].corr(method='pearson')
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                        title="Numeric Association Heatmap (Pearson's r)",
                        color_continuous_scale='RdBu_r')
        fig.show()
