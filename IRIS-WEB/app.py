import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Load the trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Species mapping
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests"""
    if request.method == 'POST':
        try:
            # Get form data
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            
            # Create input array
            input_data = np.array([[sepal_length, sepal_width, 
                                   petal_length, petal_width]])
            
            # Make prediction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            
            species = species_mapping[prediction[0]]
            confidence = round(np.max(probability) * 100, 2)
            
            # Get probabilities for each species
            prob_dict = {}
            for i in range(3):
                prob_dict[species_mapping[i]] = round(probability[0][i] * 100, 2)
            
            return render_template('predict.html', 
                                 prediction=species,
                                 confidence=confidence,
                                 probabilities=prob_dict,
                                 input_data={
                                     'sepal_length': sepal_length,
                                     'sepal_width': sepal_width,
                                     'petal_length': petal_length,
                                     'petal_width': petal_width
                                 })
            
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/eda')
def eda():
    """Generate and display EDA visualizations"""
    try:
        # Load iris data
        iris = pd.read_csv("Datasets/Iris.csv")
        iris['Species'] = iris['Species'].str.replace('Iris-', '', regex=False)
        
        # Create visualizations
        plots = {}
        
        # 1. Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, feature in enumerate(features):
            for species in iris['Species'].unique():
                species_data = iris[iris['Species'] == species][feature]
                axes[i].hist(species_data, alpha=0.5, label=species, 
                           color=colors[i % len(colors)], bins=20)
            axes[i].set_title(f'{feature} Distribution', fontweight='bold')
            axes[i].set_xlabel('cm')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        dist_plot = fig_to_base64(fig)
        plots['distributions'] = dist_plot
        plt.close(fig)
        
        # 2. Box plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            sns.boxplot(x='Species', y=feature, data=iris, ax=axes[i], 
                       palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[i].set_title(f'{feature} by Species', fontweight='bold')
            axes[i].set_ylabel('cm')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        box_plot = fig_to_base64(fig)
        plots['boxplots'] = box_plot
        plt.close(fig)
        
        # 3. Pair plot
        fig = plt.figure(figsize=(12, 10))
        species_colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
        
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i != j:
                    ax = plt.subplot(4, 4, i*4 + j + 1)
                    for species, color in species_colors.items():
                        species_data = iris[iris['Species'] == species]
                        ax.scatter(species_data[feature1], species_data[feature2], 
                                 alpha=0.6, color=color, label=species, s=30)
                    if i == 3:
                        ax.set_xlabel(feature2)
                    if j == 0:
                        ax.set_ylabel(feature1)
        
        plt.suptitle('Feature Relationships', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pair_plot = fig_to_base64(fig)
        plots['pairplot'] = pair_plot
        plt.close(fig)
        
        # 4. Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_df = iris.select_dtypes(include=[np.number])
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, ax=ax, cbar_kws={"shrink": .8})
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        heatmap_plot = fig_to_base64(fig)
        plots['heatmap'] = heatmap_plot
        plt.close(fig)
        
        # 5. Statistics table
        stats = iris.groupby('Species').agg({
            'SepalLengthCm': ['mean', 'std', 'min', 'max'],
            'SepalWidthCm': ['mean', 'std', 'min', 'max'],
            'PetalLengthCm': ['mean', 'std', 'min', 'max'],
            'PetalWidthCm': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        return render_template('eda.html', plots=plots, stats=stats.to_html(classes='table table-striped'))
        
    except Exception as e:
        return render_template('eda.html', error=str(e))

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        input_data = np.array([[data['sepal_length'], data['sepal_width'],
                              data['petal_length'], data['petal_width']]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return jsonify({
            'species': species_mapping[prediction],
            'probabilities': {
                'setosa': float(probability[0]),
                'versicolor': float(probability[1]),
                'virginica': float(probability[2])
            },
            'confidence': float(np.max(probability))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)