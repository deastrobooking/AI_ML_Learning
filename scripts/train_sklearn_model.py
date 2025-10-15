"""
Example script for training on tabular data using scikit-learn
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def parse_args():
    parser = argparse.ArgumentParser(description='Train tabular classification model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--target_col', type=str, required=True, help='Name of target column')
    parser.add_argument('--model', type=str, default='random_forest', 
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       help='Model type')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--save_dir', type=str, default='./models/saved', help='Directory to save model')
    parser.add_argument('--experiment_name', type=str, default='tabular_model', help='Experiment name')
    return parser.parse_args()


def get_model(model_name: str):
    """Get scikit-learn model by name"""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    return models[model_name]


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Tabular Data Classification Training")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Prepare features and target
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\nTraining {args.model}...")
    model = get_model(args.model)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f'{args.experiment_name}_model.pkl')
    scaler_path = os.path.join(args.save_dir, f'{args.experiment_name}_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
