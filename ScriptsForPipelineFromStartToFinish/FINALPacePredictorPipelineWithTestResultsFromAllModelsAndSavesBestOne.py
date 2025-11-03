import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os
import pickle
import json
    
class PacePrediction:
    def __init__(self, random_state=42, tune_hyperparameters=False):
        self.random_state = random_state
        self.tune_hyperparameters = tune_hyperparameters
        self.scaler = StandardScaler()
        
        if tune_hyperparameters:
            self.models = {}
            self.param_grids = {
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Ridge': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
                },
                'LinearRegression': {
                }
            }
        else:
            self.models = {
                'XGBoost': XGBRegressor(random_state=random_state, n_estimators=100),
                'RandomForest': RandomForestRegressor(random_state=random_state, n_estimators=100),
                'Ridge': Ridge(alpha=1.0),
                'LinearRegression': LinearRegression()
            }
        
        self.trained_models = {}
        self.train_results = {}
        self.val_results = {}
        self.test_results = {}
        self.best_params = {}
    
    def _create_train_val_test_split(self, X, y, groups, train_size=0.6, val_size=0.2, test_size=0.2):
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0")
        
        
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state) # separate test set
        temp_idx, test_idx = next(gss_test.split(X, y, groups))
        
        
        temp_size = train_size + val_size # separate train and validation from remaining data
        val_size_adjusted = val_size / temp_size
        
        X_temp, y_temp, groups_temp = X.iloc[temp_idx], y.iloc[temp_idx], groups.iloc[temp_idx]
        
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=self.random_state)
        train_idx_temp, val_idx_temp = next(gss_val.split(X_temp, y_temp, groups_temp))
        
        train_idx = temp_idx[train_idx_temp]
        val_idx = temp_idx[val_idx_temp]
        
        return train_idx, val_idx, test_idx
    
    def train_and_evaluate(self, historic_data_path, train_size=0.6, val_size=0.2, test_size=0.2, cv_folds=5):

        df = pd.read_csv(historic_data_path)
        
        feature_cols = [col for col in df.columns if col not in ['avg_pace_min/km', 'run_id']]
        X = df[feature_cols]
        y = df['avg_pace_min/km']
        groups = df['run_id']
        
        print(f"Dataset: {len(df)} segments from {df['run_id'].nunique()} runs")
        print(f"Features: {feature_cols}")
        
        
        print(f"\ntrain/validation/test split ({int(train_size*100)}%/{int(val_size*100)}%/{int(test_size*100)}%)...") # train/validation/test split
        train_idx, val_idx, test_idx = self._create_train_val_test_split(
            X, y, groups, train_size, val_size, test_size
        )
        
        X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
        y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_val = groups.iloc[val_idx]
        groups_test = groups.iloc[test_idx]
        
        print(f"Train set: {len(X_train)} segments from {groups_train.nunique()} runs")
        print(f"Validation set: {len(X_val)} segments from {groups_val.nunique()} runs")
        print(f"Test set: {len(X_test)} segments from {groups_test.nunique()} runs")
        
        
        X_train_scaled = self.scaler.fit_transform(X_train) # scale features
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        
        print(f"\nUsing GroupKFold with {cv_folds} folds to keep run segments together...") # Cross-validation setup for proper evaluation
        group_kfold = GroupKFold(n_splits=cv_folds)
        
        print(f"Training set has {groups_train.nunique()} unique runs")
        
        
        for name in ['XGBoost', 'RandomForest', 'Ridge', 'LinearRegression']: # train and evaluate each model
            print(f"Training {name}.")
            
            
            if name == 'XGBoost': # training and cross-validation on training set
                base_model = XGBRegressor(random_state=self.random_state)
            elif name == 'RandomForest':
                base_model = RandomForestRegressor(random_state=self.random_state)
            elif name == 'Ridge':
                base_model = Ridge()
            elif name == 'LinearRegression':
                base_model = LinearRegression()
            
            
            cv_scores = cross_val_score( # baseline CV performance
                base_model, X_train_scaled, y_train,
                groups=groups_train,
                cv=group_kfold,
                scoring='neg_mean_squared_error'
            )
            rmse_scores = np.sqrt(-cv_scores)
            print(f"Baseline CV RMSE: {rmse_scores.mean():.4f} +- {rmse_scores.std():.4f}")
            
            
            if self.tune_hyperparameters and self.param_grids[name]: # hyperparameter tuning using GroupKFold cross-validation on training set
                print(f"Tuning hyperparameters")
                
                
                grid_search = GridSearchCV( # GroupKFold for hyperparameter tuning
                    base_model, 
                    self.param_grids[name],
                    cv=GroupKFold(n_splits=cv_folds),
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                
                grid_search.fit(X_train_scaled, y_train, groups=groups_train) # groups keep runs together
                
                best_model = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                
                print(f"Best parameters: {self.best_params[name]}")
                print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
                
                
                val_pred = best_model.predict(X_val_scaled) # evaluate best model on validation set
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                print(f"Validation RMSE with best params: {val_rmse:.4f}")
                
            elif name == 'LinearRegression':
                best_model = LinearRegression()
                best_model.fit(X_train_scaled, y_train)
                print("No hyperparameters to tune for Linear Regression")
            else:
                if name == 'XGBoost': # default parameters
                    best_model = XGBRegressor(random_state=self.random_state, n_estimators=100)
                elif name == 'RandomForest':
                    best_model = RandomForestRegressor(random_state=self.random_state, n_estimators=100)
                elif name == 'Ridge':
                    best_model = Ridge(alpha=1.0)
                
                best_model.fit(X_train_scaled, y_train)
                print("Using default parameters")
            
            
            print(f"Final GroupKFold CV evaluation with best model: ") # final cross-validation evaluation with best model using GroupKFold
            final_cv_scores = cross_val_score(
                best_model, X_train_scaled, y_train,
                groups=groups_train,
                cv=group_kfold,
                scoring='neg_mean_squared_error'
            )
            final_rmse_scores = np.sqrt(-final_cv_scores)
            print(f"Final CV RMSE: {final_rmse_scores.mean():.4f} +- {final_rmse_scores.std():.4f}")
            
            
            self.trained_models[name] = best_model # save the best model
            
            train_pred = best_model.predict(X_train_scaled)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)
            
            self.train_results[name] = {
                'mae': train_mae,
                'rmse': train_rmse,
                'r2': train_r2
            }
            
            
            val_pred = best_model.predict(X_val_scaled) # validation set evaluation
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            self.val_results[name] = {
                'mae': val_mae,
                'rmse': val_rmse,
                'r2': val_r2
            }
            
            
            test_pred = best_model.predict(X_test_scaled) # test set evaluation
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)
            
            self.test_results[name] = {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2
            }
            
            print(f"\nResults for {name}:") # model results
            print(f"  Training   - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
            print(f"  Validation - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
            print(f"  Test       - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
        
        self.feature_cols = feature_cols
        
        print(f"FINAL MODEL COMPARISON:")
        
        print(f"\nTraining Set Results:")
        print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
        print("-" * 40)
        for name, results in self.train_results.items():
            print(f"{name:<15} {results['mae']:<8.4f} {results['rmse']:<8.4f} {results['r2']:<8.4f}")
        
        print(f"\nValidation Set Results:")
        print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
        print("-" * 40)
        for name, results in self.val_results.items():
            print(f"{name:<15} {results['mae']:<8.4f} {results['rmse']:<8.4f} {results['r2']:<8.4f}")
        
        print(f"\nTest Set Results (Final Performance):")
        print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
        print("-" * 40)
        for name, results in self.test_results.items():
            print(f"{name:<15} {results['mae']:<8.4f} {results['rmse']:<8.4f} {results['r2']:<8.4f}")
        
        
        best_mae_model = min(self.test_results.keys(), key=lambda x: self.test_results[x]['mae']) # highlight best models
        best_rmse_model = min(self.test_results.keys(), key=lambda x: self.test_results[x]['rmse'])
        best_r2_model = max(self.test_results.keys(), key=lambda x: self.test_results[x]['r2'])
        
        print(f"\nBest Models (Test Set):")
        print(f"  Best MAE:  {best_mae_model} ({self.test_results[best_mae_model]['mae']:.4f})")
        print(f"  Best RMSE: {best_rmse_model} ({self.test_results[best_rmse_model]['rmse']:.4f})")
        print(f"  Best R2:   {best_r2_model} ({self.test_results[best_r2_model]['r2']:.4f})")
        
        self._plot_all_results()
        self._plot_predictions_vs_actual(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
        
        self._plot_test_run_example(df, test_idx)
    
    def predict_new_route(self, new_route_path, output_path=None):
        print(f"\nLoading new route data")
        new_df = pd.read_csv(new_route_path)
        
        X_new = new_df[self.feature_cols]
        X_new_scaled = self.scaler.transform(X_new)
        
        print(f"New route: {len(new_df)} segments")
        
        
        output_df = new_df.copy() # a copy of the original csv to append predictions
        predictions = {}
        
        for name, model in self.trained_models.items():
            pred = model.predict(X_new_scaled)
            predictions[name] = pred
            
            output_df[f'predicted_pace_{name}_min_per_km'] = pred
            
            total_time = pred.sum()
            avg_pace = pred.mean()
            print(f"\n{name} predictions:")
            print(f"  Average pace: {avg_pace:.2f} min/km")
            print(f"  Total time: {total_time:.1f} min ({int(total_time//60)}:{int(total_time%60):02d})")
            print(f"  Pace range: {pred.min():.2f} to {pred.max():.2f} min/km")
            print(f"  Test MAE: {self.test_results[name]['mae']:.4f} min/km")
            

        model_columns = [f'predicted_pace_{name}_min_per_km' for name in self.trained_models.keys()]
        output_df['predicted_pace_mean_min_per_km'] = output_df[model_columns].mean(axis=1)
        output_df['predicted_pace_std_min_per_km'] = output_df[model_columns].std(axis=1)

        if output_path is None:
            input_dir = os.path.dirname(new_route_path)
            input_filename = os.path.basename(new_route_path)
            input_name, input_ext = os.path.splitext(input_filename)
            output_path = os.path.join(input_dir, f"{input_name}_with_predictions{input_ext}")
        
        output_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        new_columns = [col for col in output_df.columns if col not in new_df.columns]
        print(f"Added columns: {new_columns}")
        
        print(f"\Head of predictions:")
        preview_columns = ['segment_km'] + model_columns + ['predicted_pace_mean_min_per_km', 'predicted_pace_std_min_per_km']
        if 'segment_km' in output_df.columns:
            print(output_df[preview_columns].head().round(3))
        else:
            print(output_df[model_columns + ['predicted_pace_mean_min_per_km', 'predicted_pace_std_min_per_km']].head().round(3))
        
        self._plot_new_route_predictions(new_df, predictions)
        return predictions, output_path
    
    def _plot_all_results(self):
        models = list(self.test_results.keys())
        
        train_mae = [self.train_results[m]['mae'] for m in models]
        val_mae = [self.val_results[m]['mae'] for m in models]
        test_mae = [self.test_results[m]['mae'] for m in models]
        
        train_rmse = [self.train_results[m]['rmse'] for m in models]
        val_rmse = [self.val_results[m]['rmse'] for m in models]
        test_rmse = [self.test_results[m]['rmse'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, train_mae, width, label='Training', alpha=0.8, color='lightblue') # MAE plot
        ax1.bar(x, val_mae, width, label='Validation', alpha=0.8, color='orange')
        ax1.bar(x + width, test_mae, width, label='Test', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('MAE (min/km)')
        ax1.set_title('Mean Absolute Error Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(x - width, train_rmse, width, label='Training', alpha=0.8, color='lightblue') # RMSE plot
        ax2.bar(x, val_rmse, width, label='Validation', alpha=0.8, color='orange')
        ax2.bar(x + width, test_rmse, width, label='Test', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE (min/km)')
        ax2.set_title('Root Mean Squared Error Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_predictions_vs_actual(self, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test):
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        
        set_names = ['Training', 'Validation', 'Test']
        X_sets = [X_train_scaled, X_val_scaled, X_test_scaled]
        y_sets = [y_train, y_val, y_test]
        colors = ['lightblue', 'orange', 'lightcoral']
        
        for i, (name, model) in enumerate(self.trained_models.items()):
            for j, (set_name, X_set, y_set, color) in enumerate(zip(set_names, X_sets, y_sets, colors)):
                pred = model.predict(X_set)
                
                axes[i, j].scatter(y_set, pred, alpha=0.6, color=color)
                axes[i, j].plot([y_set.min(), y_set.max()], [y_set.min(), y_set.max()], 'r--', lw=2)
                
                if j == 2:  # test set
                    mae = self.test_results[name]['mae']
                    r2 = self.test_results[name]['r2']
                elif j == 1:  # validation set
                    mae = self.val_results[name]['mae']
                    r2 = self.val_results[name]['r2']
                else:  # training set
                    mae = self.train_results[name]['mae']
                    r2 = self.train_results[name]['r2']
                
                axes[i, j].set_xlabel('Actual Pace (min/km)')
                axes[i, j].set_ylabel('Predicted Pace (min/km)')
                axes[i, j].set_title(f'{name} - {set_name}\nMAE: {mae:.4f}, R2: {r2:.4f}')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_test_run_example(self, df, test_idx):
       
        test_df = df.iloc[test_idx].copy() # test set data
        test_runs = test_df['run_id'].unique()
        
        
        run_segments_count = test_df['run_id'].value_counts() # random test run 
        runs_with_multiple_segments = run_segments_count[run_segments_count >= 3].index
        
        if len(runs_with_multiple_segments) > 0:
            selected_run_id = np.random.choice(runs_with_multiple_segments)
        else:
            selected_run_id = np.random.choice(test_runs)
        
        run_data = test_df[test_df['run_id'] == selected_run_id].copy()
        run_data = run_data.sort_values('segment_km') if 'segment_km' in run_data.columns else run_data.reset_index(drop=True)
        
        print(f"TEST SET EXAMPLE:")
        print(f"Selected test run ID: {selected_run_id}")
        print(f"Number of segments: {len(run_data)}")
        
        X_run = run_data[self.feature_cols]
        X_run_scaled = self.scaler.transform(X_run)
        y_actual = run_data['avg_pace_min/km']
        
        run_predictions = {} # predictions from all models
        model_maes = {}
        
        for name, model in self.trained_models.items():
            pred = model.predict(X_run_scaled)
            run_predictions[name] = pred
            
            
            run_mae = mean_absolute_error(y_actual, pred) # MAE for this run
            model_maes[name] = run_mae
            
            print(f"\n{name} for this run:")
            print(f"  Actual avg pace: {y_actual.mean():.2f} min/km")
            print(f"  Predicted avg pace: {pred.mean():.2f} min/km")
            print(f"  MAE for this run: {run_mae:.4f} min/km")
            print(f"  Overall test MAE: {self.test_results[name]['mae']:.4f} min/km")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if 'segment_km' in run_data.columns:
            segments = run_data['segment_km']
            xlabel = 'Segment (km)'
        else:
            segments = range(len(run_data))
            xlabel = 'Segment Index'
        
        colors = ['blue', 'orange', 'green', 'red']
        
        
        ax1.plot(segments, y_actual, marker='o', linewidth=3, color='black', # actual pace, thick black line
                label=f'Actual Pace', markersize=8, zorder=5)
        

        for i, (model_name, paces) in enumerate(run_predictions.items()):
            ax1.plot(segments, paces, marker='s', label=f'{model_name}', 
                    linewidth=2, color=colors[i % len(colors)], alpha=0.8, markersize=6)

        ax1.axhspan(3, 8, alpha=0.1, color='green', label='Normal pace range', zorder=1)
        
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Pace (min/km)')
        ax1.set_title(f'Test Run Validation: Actual vs Predicted Paces\nRun ID: {selected_run_id} ({len(run_data)} segments)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        if 'elevation_gain_m' in run_data.columns and 'elevation_loss_m' in run_data.columns:
            net_elevation = run_data['elevation_gain_m'] - run_data['elevation_loss_m']
            bars = ax2.bar(segments, net_elevation, alpha=0.7, color='green', label='Net Elevation')
            ax2.set_ylabel('Net Elevation (m)', color='green')
            ax2.set_title(f'Net Elevation Profile for Test Run {selected_run_id}')
        elif 'elevation_gain_m' in run_data.columns:
            bars = ax2.bar(segments, run_data['elevation_gain_m'], alpha=0.7, color='green', label='Elevation Gain')
            ax2.set_ylabel('Elevation Gain (m)', color='green')
            ax2.set_title(f'Elevation Profile for Test Run {selected_run_id}')
            print("Warning: elevation_loss_m column not found, using elevation_gain_m only")
        
        ax2.set_xlabel(xlabel)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nDetailed Segment Analysis for Test Run {selected_run_id}:")
        header = "Seg"
        if 'elevation_gain_m' in run_data.columns and 'elevation_loss_m' in run_data.columns:
            header += " | Net Elev"
        elif 'elevation_gain_m' in run_data.columns:
            header += " | Elev Gain"
        header += " | Actual | XGBoost | RandForest | Ridge | LinearReg | Best Model"
        print(header)
        print("-" * len(header))
        
        for i, (idx, row) in enumerate(run_data.iterrows()):
            if 'segment_km' in run_data.columns:
                line = f" {int(row['segment_km']):2d} "
            else:
                line = f" {i:2d} "
            
            if 'elevation_gain_m' in run_data.columns and 'elevation_loss_m' in run_data.columns:
                net_elev = row['elevation_gain_m'] - row['elevation_loss_m']
                line += f"| {net_elev:+5.0f}m"
            elif 'elevation_gain_m' in run_data.columns:
                line += f"| {row['elevation_gain_m']:4.0f}m"
            
            actual_pace = row['avg_pace_min/km']
            line += f"| {actual_pace:6.2f} "
            
            segment_errors = {}
            for model_name in ['XGBoost', 'RandomForest', 'Ridge', 'LinearRegression']:
                pred_pace = run_predictions[model_name][i]
                error = abs(actual_pace - pred_pace)
                segment_errors[model_name] = error
                line += f"| {pred_pace:7.2f} "
            
            best_model = min(segment_errors, key=segment_errors.get) # best model for this segment
            line += f"| {best_model}"
            
            print(line)
        
        print(f"\nSummary for Test Run {selected_run_id}:")
        print(f"Actual run statistics:")
        print(f"  Total time: {y_actual.sum():.1f} min ({int(y_actual.sum()//60)}:{int(y_actual.sum()%60):02d})")
        print(f"  Average pace: {y_actual.mean():.2f} min/km")
        print(f"  Pace range: {y_actual.min():.2f} to {y_actual.max():.2f} min/km")
        
        best_model_for_run = min(model_maes, key=model_maes.get)
        print(f"\nBest performing model for this run: {best_model_for_run}")
        print(f"  MAE: {model_maes[best_model_for_run]:.4f} min/km")
        print(f"  Average predicted pace: {run_predictions[best_model_for_run].mean():.2f} min/km")
        
        return selected_run_id, run_data, run_predictions, y_actual

    def _plot_new_route_predictions(self, route_df, predictions):

        print(f"NEW ROUTE PACE PREDICTIONS:")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if 'segment_km' in route_df.columns:
            segments = route_df['segment_km']
            xlabel = 'Segment (km)'
        else:
            segments = range(len(route_df))
            xlabel = 'Segment Index'
        
        
        colors = ['blue', 'orange', 'green', 'red'] # pace predictions
        for i, (model_name, paces) in enumerate(predictions.items()):
            ax1.plot(segments, paces, marker='o', label=f'{model_name}', 
                    linewidth=2, color=colors[i % len(colors)])
        
        ax1.axhspan(3, 8, alpha=0.1, color='green', label='Normal pace range')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Predicted Pace (min/km)')
        ax1.set_title('NEW ROUTE: Pace Predictions by Segment')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns: # net elevation profile
            net_elevation = route_df['elevation_gain_m'] - route_df['elevation_loss_m']
            ax2.bar(segments, net_elevation, alpha=0.7, color='green', label='Net Elevation')
            ax2.set_ylabel('Net Elevation (m)', color='green')
            ax2.set_title('NEW ROUTE: Net Elevation Profile')
        elif 'elevation_gain_m' in route_df.columns:

            ax2.bar(segments, route_df['elevation_gain_m'], alpha=0.7, color='green', label='Elevation Gain')
            ax2.set_ylabel('Elevation Gain (m)', color='green')
            ax2.set_title('NEW ROUTE: Elevation Profile')
        
        ax2.set_xlabel(xlabel)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nDetailed Segment Predictions:")
        header = "Segment"
        if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns:
            header += " | Net Elev "
        elif 'elevation_gain_m' in route_df.columns:
            header += " | Elev Gain"
        header += " | XGBoost | RandomForest | Ridge   | LinearReg"
        print(header)
        print("-" * len(header))
        
        for i in range(len(route_df)):
            seg = route_df.iloc[i]
            if 'segment_km' in route_df.columns:
                row = f"  {int(seg['segment_km']):2d}    "
            else:
                row = f"  {i:2d}    "
            
            if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns:
                net_elev = seg['elevation_gain_m'] - seg['elevation_loss_m']
                row += f"| {net_elev:+5.0f}m   "
            elif 'elevation_gain_m' in route_df.columns:
                row += f"|   {seg['elevation_gain_m']:3.0f}m    "
            
            row += f"| {predictions['XGBoost'][i]:6.2f}  |    {predictions['RandomForest'][i]:6.2f}   | "
            row += f"{predictions['Ridge'][i]:6.2f}  |   {predictions['LinearRegression'][i]:6.2f}"
            print(row)
    
    def get_test_mae_results(self):
        return {name: results['mae'] for name, results in self.test_results.items()}
    
    def get_all_results(self):
        return {
            'train': self.train_results,
            'validation': self.val_results,
            'test': self.test_results,
            'best_params': self.best_params
        }

    def save_best_model(self, output_dir, model_name='LinearRegression'):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        
        model_path = os.path.join(output_dir, f'{model_name}_model.pkl') # saving the trained model
        with open(model_path, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)
        
        
        scaler_path = os.path.join(output_dir, f'{model_name}_scaler.pkl') # save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        
        metadata = { # feature columns and metadata
            'model_name': model_name,
            'feature_cols': self.feature_cols,
            'test_mae': self.test_results[model_name]['mae'],
            'test_rmse': self.test_results[model_name]['rmse'],
            'test_r2': self.test_results[model_name]['r2'],
            'best_params': self.best_params.get(model_name, {})
        }
        
        metadata_path = os.path.join(output_dir, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved successfully to {output_dir}")
        print(f"Files created:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"  - {metadata_path}")
        
        return output_dir

def train_models(historic_csv_path, train_size=0.6, val_size=0.2, test_size=0.2, 
                cv_folds=5, tune_hyperparameters=False):
    predictor = PacePrediction(tune_hyperparameters=tune_hyperparameters)
    predictor.train_and_evaluate(historic_csv_path, train_size, val_size, test_size, cv_folds)
    return predictor

def predict_pace(trained_predictor, new_route_csv_path, output_csv_path=None):
    return trained_predictor.predict_new_route(new_route_csv_path, output_csv_path)

if __name__ == "__main__":
    print("Pace Prediction with Train/Validation/Test Split")
    
    historic_csv = r'C:\Users\User\Desktop\SkripsieWebAppRepo\running-pace-app\FinalDataset\LatestDataset_Cleaned_Removed_Anomaly_Runs_With_All_Features.csv'
    new_route_csv = r'C:\Users\User\Desktop\TestingIncompleteSegment.csv'
    
    output_csv = None
    
    print("\nTraining all models with train/validation/test split")
    predictor = train_models(
        historic_csv, 
        train_size=0.8,      # 80% for training
        val_size=0.1,        # 10% for validation
        test_size=0.1,       # 10% for test
        cv_folds=5, 
        tune_hyperparameters=True
    )
    
    print("\nGetting final test results")
    test_mae_results = predictor.get_test_mae_results()
    all_results = predictor.get_all_results()

    
    print("\nLinear Regression coefficients") # Linear Regression coefficients
    lr_model = predictor.trained_models['LinearRegression']
    feature_names = predictor.feature_cols
    coefficients = lr_model.coef_
    
    print("\nLinear Regression Feature Coefficients:")
    print(f"{'Feature':<30} {'Coefficient':>15} {'Impact':>20}")
    
    coef_pairs = sorted(zip(feature_names, coefficients), 
                       key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef in coef_pairs:
        impact = "Slower pace" if coef > 0 else "Faster pace"
        print(f"{feature:<30} {coef:>15.6f}   {impact:>20}")
    
    print(f"{'Intercept':<30} {lr_model.intercept_:>15.6f}")

    print("\nFinal Test MAE Results:")
    for model, mae in test_mae_results.items():
        print(f"{model}: {mae:.4f} min/km")
    
    print("\n3. Predicting pace for new route:")
    predictions, output_path = predict_pace(predictor, new_route_csv, output_csv)
    
    print(f"\nResults saved to: {output_path}")

    # Save the best model
    #print("\n3. Saving the best model (Linear Regression)...")
    #model_dir = r'C:\\Users\\User\\Desktop\\SkripsieWebAppRepo\\Saved_Models'
    #predictor.save_best_model(model_dir, 'LinearRegression')