import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import InputFile  # Moved to separate import for clarity
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io
from datetime import datetime
import seaborn as sns

    # --- Enhanced Configuration ---
TOKEN = "7781005960:AAEYxCF3hnhJ0a4QYTfoxS-VbBCE1ErMPZ0"
PATTERN_WINDOWS = [3, 5, 7, 10]
MIN_STREAK_LENGTH = 2
CONFIDENCE_THRESHOLD = 0.90  # Changed from 0.75 to 0.90
TRAIN_INTERVAL = 100
MODEL_ENSEMBLE = True
USE_XGBOOST = True
ENABLE_HYPERPARAM_TUNING = True  # Enable grid search hyperparameter tuning
ADAPTIVE_RETRAIN_THRESHOLD = 0.70  # Retrain if accuracy falls below this

    # --- Pattern Definitions ---
KNOWN_PATTERNS = {
        'Single Trend': ['B', 'S', 'B', 'S', 'B'],
        'Double Trend': ['S', 'S', 'B', 'S', 'S'],
        'Triple Trend': ['B', 'B', 'B', 'S', 'S'],
        'Quadra Trend': ['S', 'S', 'S', 'B', 'B', 'B'],
        'Three in One': ['B', 'B', 'B', 'S', 'B', 'B'],
        'Two in One': ['S', 'S', 'B', 'S', 'S', 'B', 'S', 'S'],
        'Three in Two': ['B', 'B', 'B', 'S', 'S', 'B', 'B'],
        'Four in One': ['S', 'S', 'S', 'S', 'B', 'S', 'S', 'S'],
        'Four in Two': ['B', 'B', 'B', 'S', 'S', 'B', 'B', 'B'],
        'Long Trend': ['S']*9
    }

class EnhancedPatternAnalyzer:
        def __init__(self):
            self.history = deque(maxlen=1000)
            self.pattern_counts = defaultdict(int)
            self.streaks = {'B': 0, 'S': 0}
            self.last_prediction = None
            self.prediction_history = []
            self.known_patterns = KNOWN_PATTERNS
            self.streak_history = [(0, 0)]
            
        def add_result(self, result: str):
            result = result.upper()
            if result not in ['B', 'S']:
                return False
                
            prev_result = self.history[-1] if self.history else None
            self.history.append(result)
            
            if prev_result == result:
                self.streaks[result] += 1
            else:
                self.streaks[result] = 1
                self.streaks['B' if result == 'S' else 'S'] = 0
            
            self.streak_history.append((self.streaks['B'], self.streaks['S']))
            
            # Update pattern counts
            for window in PATTERN_WINDOWS:
                if len(self.history) >= window:
                    pattern = ''.join(list(self.history)[-window:])
                    self.pattern_counts[pattern] += 1
                    
            return True

        def detect_known_patterns(self):
            detected = []
            if not self.history:
                return detected
                
            history_str = ''.join(self.history)
            
            for name, pattern in self.known_patterns.items():
                pattern_str = ''.join(pattern)
                if len(history_str) < len(pattern_str):
                    continue
                    
                if history_str.endswith(pattern_str):
                    # Determine next outcome logic
                    if name == 'Single Trend':
                        next_outcome = 'S' if pattern[-1] == 'B' else 'B'
                    elif name == 'Double Trend':
                        next_outcome = 'B' if pattern[-1] == 'S' else 'S'
                    elif 'Three' in name or 'Quadra' in name or 'Four' in name:
                        next_outcome = 'S' if pattern[-1] == 'B' else 'B'
                    elif 'Long Trend' in name:
                        next_outcome = 'B' if pattern[-1] == 'S' else 'S'
                    else:
                        next_outcome = 'B'
                    
                    confidence = 0.85
                    detected.append((next_outcome, confidence, f"Known pattern: {name}"))
            
            return detected

        def get_features(self):
            if not self.history:
                return None
                
            features = {
                'big_ratio': sum(1 for x in self.history if x == 'B') / len(self.history),
                'current_streak_B': self.streaks['B'],
                'current_streak_S': self.streaks['S'],
                'last_3': ''.join(list(self.history)[-3:]) if len(self.history) >= 3 else '',
                'last_5': ''.join(list(self.history)[-5:]) if len(self.history) >= 5 else '',
                'last_7': ''.join(list(self.history)[-7:]) if len(self.history) >= 7 else '',
                'last_10': ''.join(list(self.history)[-10:]) if len(self.history) >= 10 else '',
                'last_pred_correct': 1 if self.last_prediction == self.history[-1] else 0,
                'big_last_10': sum(1 for x in list(self.history)[-10:]) if len(self.history) >= 10 else 0.5,
                'streak_ratio': self.streaks['B'] / (self.streaks['B'] + self.streaks['S'] + 1e-6),
                'big_last_20': sum(1 for x in list(self.history)[-20:])/20 if len(self.history) >= 20 else 0.5,
                'transition_count': sum(1 for i in range(1, len(self.history)) if self.history[i] != self.history[i-1]),
                'last_3_big': sum(1 for x in list(self.history)[-3:] if x == 'B') if len(self.history) >=3 else 0,
                'last_5_big': sum(1 for x in list(self.history)[-5:] if x == 'B') if len(self.history) >= 5 else 0
            }
            
            for window in PATTERN_WINDOWS:
                if len(self.history) >= window:
                    pattern = ''.join(list(self.history)[-window:])
                    features[f'pattern_{window}_freq'] = self.pattern_counts.get(pattern, 0)
                    
            if len(self.streak_history) > 5:
                last_streaks = np.array(self.streak_history[-5:])
                features['streak_momentum_B'] = np.mean(last_streaks[:, 0])
                features['streak_momentum_S'] = np.mean(last_streaks[:, 1])
            else:
                features['streak_momentum_B'] = 0
                features['streak_momentum_S'] = 0
                
            # Add new lag and moving average features
            history_arr = np.array([1 if x == 'B' else 0 for x in self.history])
            if len(history_arr) >= 5:
                features['ma_3'] = np.mean(history_arr[-3:])
                features['ma_5'] = np.mean(history_arr[-5:])
            else:
                features['ma_3'] = 0.5
                features['ma_5'] = 0.5
            if len(history_arr) >= 2:
                features['lag_1'] = history_arr[-2]
            else:
                features['lag_1'] = 0.5

            return features

        def predict_next(self) -> list:
            predictions = self.detect_known_patterns()
            
            # Streak based predictions
            for outcome in ['B', 'S']:
                if self.streaks[outcome] >= MIN_STREAK_LENGTH:
                    streak_length = self.streaks[outcome]
                    reversal_prob = min(0.95, 0.6 + streak_length * 0.05)
                    opposite = 'S' if outcome == 'B' else 'B'
                    predictions.append(
                        (opposite, reversal_prob, f"Streak reversal after {streak_length} {outcome}s (prob: {reversal_prob:.2f})")
                    )
            
            # Pattern frequency analysis
            for window in sorted(PATTERN_WINDOWS, reverse=True):
                if len(self.history) >= window:
                    recent_pattern = ''.join(list(self.history)[-window:])
                    count = self.pattern_counts.get(recent_pattern, 0)
                    if count > 0:
                        confidence = min(0.85, 0.5 + count * 0.05)
                        next_outcome = self._get_pattern_followup(recent_pattern, window)
                        predictions.append(
                            (next_outcome, confidence, f"Pattern {recent_pattern} seen {count}x (window={window})")
                        )
            
            # Fallback probability with momentum
            if len(self.history) > 0:
                big_count = sum(1 for x in self.history if x == 'B')
                total = len(self.history)
                big_prob = (big_count + 1) / (total + 2)
                last_20 = list(self.history)[-20:] if len(self.history) >= 20 else list(self.history)
                weights = [0.9 ** i for i in range(len(last_20))][::-1]
                weighted_b = sum(w * (1 if x=='B' else 0) for w,x in zip(weights, last_20))
                weighted_total = sum(weights)
                recent_bias = weighted_b / weighted_total
                streak_factor = 1.0
                if len(self.streak_history) > 3:
                    last_streaks = np.array(self.streak_history[-3:])
                    avg_streak_B = np.mean(last_streaks[:, 0])
                    avg_streak_S = np.mean(last_streaks[:, 1])
                    streak_factor = 1.0 + (avg_streak_B - avg_streak_S) * 0.05
                adjusted_prob = (0.5 * big_prob + 0.5 * recent_bias) * streak_factor
                adjusted_prob = max(0.1, min(0.9, adjusted_prob))
                predictions.append(
                    ('B' if adjusted_prob > 0.5 else 'S',
                    max(adjusted_prob, 1-adjusted_prob),
                    f"Probability (B:{adjusted_prob:.2f}, Recent:{recent_bias:.2f}, StreakFactor:{streak_factor:.2f})")
                )
            
            return predictions

        def _get_pattern_followup(self, pattern, window_size):
            if len(self.history) < window_size + 5:
                return 'B' if np.mean([1 if x=='B' else 0 for x in pattern]) > 0.5 else 'S'
            follow_b = 0
            follow_s = 0
            pattern_str = ''.join(pattern)
            for i in range(len(self.history) - window_size):
                current_window = ''.join(list(self.history)[i:i+window_size])
                if current_window == pattern_str:
                    next_outcome = self.history[i+window_size]
                    if next_outcome == 'B':
                        follow_b += 1
                    else:
                        follow_s += 1
            if follow_b + follow_s > 0:
                return 'B' if follow_b > follow_s else 'S'
            else:
                return 'B' if np.mean([1 if x=='B' else 0 for x in pattern]) > 0.5 else 'S'


class EnhancedMLPredictor:
        def __init__(self):
            # Base models with hyperparameters (add LightGBM and CatBoost)
            base_models = []
            if USE_XGBOOST:
                base_models.append(('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, use_label_encoder=False, eval_metric='logloss')))
            base_models.extend([
                ('gbc', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5)),
                ('rfc', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=3)),
                ('lgbm', LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=5)),
                ('cat', CatBoostClassifier(verbose=0, iterations=150, learning_rate=0.1, depth=5))
            ])
            self.base_models = base_models
            
            # Stacking meta model
            self.meta_model = LogisticRegression(penalty='l2', max_iter=1000)
            
            self.stacking_model = None
            self.trained = False
            self.last_accuracies = {}
            self.feature_names = []
            
        def prepare_data(self, analyzer):
            X, y = [], []
            history = list(analyzer.history)
            if len(history) < max(PATTERN_WINDOWS) + 10:
                return np.array([]), np.array([])
            
            for i in range(max(PATTERN_WINDOWS), len(history)):
                window = history[i-max(PATTERN_WINDOWS):i]
                if i >= len(analyzer.streak_history):
                    break
                features = {
                    'big_ratio': sum(1 for x in window if x == 'B') / len(window),
                    'current_streak_B': analyzer.streak_history[i][0],
                    'current_streak_S': analyzer.streak_history[i][1],
                    'big_last_3': sum(1 for x in window[-3:]) if len(window) >= 3 else 0,
                    'big_last_5': sum(1 for x in window[-5:]) if len(window) >= 5 else 0,
                    'big_last_7': sum(1 for x in window[-7:]) if len(window) >= 7 else 0,
                    'big_last_10': sum(1 for x in window[-10:]) if len(window) >= 10 else 0,
                    'big_last_20': sum(1 for x in window[-20:])/20 if len(window) >= 20 else 0.5,
                    'transitions': sum(1 for j in range(1, len(window)) if window[j] != window[j-1]),
                    'streak_ratio': analyzer.streak_history[i][0] / (analyzer.streak_history[i][0] + analyzer.streak_history[i][1] + 1e-6),
                    'pattern_3_freq': analyzer.pattern_counts.get(''.join(window[-3:]), 0) if len(window) >= 3 else 0,
                    'pattern_5_freq': analyzer.pattern_counts.get(''.join(window[-5:]), 0) if len(window) >= 5 else 0,
                    'outcome': history[i]
                }
                
                if i > 5:
                    last_streaks = np.array(analyzer.streak_history[i-5:i])
                    features['streak_momentum_B'] = np.mean(last_streaks[:, 0])
                    features['streak_momentum_S'] = np.mean(last_streaks[:, 1])
                else:
                    features['streak_momentum_B'] = 0
                    features['streak_momentum_S'] = 0

                # Additional lag and moving average features
                window_arr = np.array([1 if x=='B' else 0 for x in window])
                if len(window_arr) >= 3:
                    features['ma_3'] = np.mean(window_arr[-3:])
                else:
                    features['ma_3'] = 0.5
                if len(window_arr) >= 5:
                    features['ma_5'] = np.mean(window_arr[-5:])
                else:
                    features['ma_5'] = 0.5
                if len(window_arr) >= 2:
                    features['lag_1'] = window_arr[-2]
                else:
                    features['lag_1'] = 0.5
                    
                X.append([
                    features['big_ratio'],
                    features['current_streak_B'],
                    features['current_streak_S'],
                    features['big_last_3']/3,
                    features['big_last_5']/5,
                    features['big_last_7']/7,
                    features['big_last_10']/10,
                    features['big_last_20'],
                    features['transitions']/len(window),
                    features['streak_ratio'],
                    features['pattern_3_freq'],
                    features['pattern_5_freq'],
                    features['streak_momentum_B'],
                    features['streak_momentum_S'],
                    features['ma_3'],
                    features['ma_5'],
                    features['lag_1']
                ])
                y.append(1 if features['outcome']=='B' else 0)
            
            self.feature_names = [
                'big_ratio', 'current_streak_B', 'current_streak_S', 
                'big_last_3', 'big_last_5', 'big_last_7', 'big_last_10',
                'big_last_20', 'transitions', 'streak_ratio',
                'pattern_3_freq', 'pattern_5_freq',
                'streak_momentum_B', 'streak_momentum_S',
                'ma_3', 'ma_5', 'lag_1'
            ]
            return np.array(X), np.array(y)

        def _apply_smote(self, X, y):
            try:
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(X, y)
                return X_res, y_res
            except Exception as e:
                logging.warning(f"SMOTE failed: {e}")
                return X, y
        
        def _hyperparameter_tuning(self, model_name, model, X, y):
            if model_name == 'xgb':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 1.0]
                }
            elif model_name == 'gbc':
                param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1]
                }
            elif model_name == 'rfc':
                param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [4, 5, 7],
                    'min_samples_split': [2, 3, 4]
                }
            elif model_name == 'lgbm':
                param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [4, 5, 7],
                    'learning_rate': [0.05, 0.1]
                }
            elif model_name == 'cat':
                param_grid = {
                    'iterations': [100, 150],
                    'depth': [4, 5, 6],
                    'learning_rate': [0.05, 0.1]
                }
            else:
                return model  # No tuning for unknown model
            
            grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
            grid_search.fit(X, y)
            logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_

        def train(self, analyzer):
            X, y = self.prepare_data(analyzer)
            if len(X) < 100:
                return {name: 0.0 for name, _ in self.base_models}
            
            # Apply SMOTE if imbalance detected (example threshold: minority class <40%)
            minority_ratio = min(np.mean(y), 1-np.mean(y))
            if minority_ratio < 0.4:
                X, y = self._apply_smote(X, y)
                logging.info("Applied SMOTE for class imbalance.")
            
            split_idx = int(len(X)*0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]
            
            trained_models = []
            accuracies = {}
            
            # Hyperparameter tuning if enabled
            for name, model in self.base_models:
                if ENABLE_HYPERPARAM_TUNING:
                    model = self._hyperparameter_tuning(name, model, X_train, y_train)
                model.fit(X_train, y_train)
                trained_models.append((name, model))
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accuracies[name] = acc
            
            # Feature selection logging (optional)
            rfc_model = RandomForestClassifier(n_estimators=100)
            rfe = RFE(rfc_model, n_features_to_select=10)
            rfe.fit(X_train, y_train)
            selected_feats = np.array(self.feature_names)[rfe.support_]
            logging.info(f"Selected features from RFE: {selected_feats}")
            
            # Train stacking model
            estimators_for_stacking = [(name, model) for name, model in trained_models]
            self.stacking_model = StackingClassifier(estimators=estimators_for_stacking, final_estimator=self.meta_model, cv=3, n_jobs=-1)
            self.stacking_model.fit(X_train, y_train)
            stack_pred = self.stacking_model.predict(X_test)
            stack_acc = accuracy_score(y_test, stack_pred)
            accuracies['stacking'] = stack_acc
            
            self.models = dict(trained_models)
            self.models['stacking'] = self.stacking_model
            self.trained = True
            self.last_accuracies = accuracies
            
            logging.info(f"Training complete. Accuracies: {accuracies}")
            return accuracies

        def predict(self, analyzer):
            if not self.trained or len(analyzer.history) < max(PATTERN_WINDOWS):
                return []
            
            features = analyzer.get_features()
            if not features:
                return []
            
            # Prepare feature vector same order as train
            feature_list = [
                features['big_ratio'],
                features['current_streak_B'],
                features['current_streak_S'],
                features['last_3'].count('B')/3 if len(features['last_3'])==3 else 0,
                features['last_5'].count('B')/5 if len(features['last_5'])==5 else 0,
                features['last_7'].count('B')/7 if len(features['last_7'])==7 else 0,
                features['last_10'].count('B')/10 if len(features['last_10'])==10 else 0,
                features['big_last_10']/10 if 'big_last_10' in features else 0,
                features['transition_count']/len(analyzer.history) if 'transition_count' in features else 0,
                features['streak_ratio'],
                features.get('pattern_3_freq', 0),
                features.get('pattern_5_freq', 0),
                features.get('streak_momentum_B', 0),
                features.get('streak_momentum_S', 0),
                features.get('ma_3', 0),
                features.get('ma_5', 0),
                features.get('lag_1', 0)
            ]
            
            X = np.array([feature_list])
            
            predictions = []
            
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba(X)[0]
                    confidence = max(proba)
                    predicted_class = 'B' if proba[1] > 0.5 else 'S'
                    predictions.append(
                        (predicted_class, confidence, f"{name.upper()} (acc: {self.last_accuracies.get(name, 0):.2f})")
                    )
                except Exception as e:
                    logging.error(f"Prediction error with {name}: {e}")
                    continue
            
            return predictions


class AdvancedWingoBot:
        def __init__(self):
            self.analyzer = EnhancedPatternAnalyzer()
            self.ml_predictor = EnhancedMLPredictor()
            self.user_sessions = {}
            self.last_training = 0
            self.accuracy_threshold = ADAPTIVE_RETRAIN_THRESHOLD
            self.last_accuracy_check = 1.0
            
        async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(
                "🎯 Advanced Wingo Predictor Bot with XGBoost, LightGBM, CatBoost, and Stacking\n\n"
                "Send Big/Small results as 'B' or 'S' (e.g., BBSSB)\n"
                "I'll provide enhanced predictions with multiple analysis methods!\n\n"
                "Features:\n"
                "- Multiple advanced ML models\n"
                "- Stacking ensemble\n"
                "- Advanced streak detection & pattern recognition\n"
                "- Feature engineering & hyperparameter tuning\n"
                "- Adaptive retraining based on model performance"
            )
            
        async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_id = update.message.from_user.id
            text = update.message.text.upper().replace(" ", "")
            
            if not text or not all(c in ['B', 'S'] for c in text):
                await update.message.reply_text("⚠️ Please only use 'B' (Big) or 'S' (Small) characters!")
                return
            
            for result in text:
                self.analyzer.add_result(result)
            
            # Adaptive retraining based on interval or performance drop
            retrain_needed = False
            if len(self.analyzer.history) - self.last_training >= TRAIN_INTERVAL:
                retrain_needed = True
            else:
                # Check if stacking accuracy dropped below threshold
                current_acc = self.ml_predictor.last_accuracies.get('stacking', 1.0)
                if current_acc < self.accuracy_threshold and current_acc < self.last_accuracy_check:
                    retrain_needed = True
            
            if retrain_needed:
                accuracies = self.ml_predictor.train(self.analyzer)
                self.last_training = len(self.analyzer.history)
                self.last_accuracy_check = max(accuracies.get('stacking', 0.0), 0.0)
                logging.info(f"Models retrained. Accuracies: {accuracies}")
            
            # Get all predictions
            pattern_preds = self.analyzer.predict_next()
            ml_preds = self.ml_predictor.predict(self.analyzer)
            all_predictions = pattern_preds + ml_preds
            
            if not all_predictions:
                await update.message.reply_text("🔍 Analyzing... Need more data for predictions")
                return
            
            valid_predictions = [p for p in all_predictions if p[1] >= CONFIDENCE_THRESHOLD]
            if valid_predictions:
                valid_predictions.sort(key=lambda x: x[1], reverse=True)
                best_pred = valid_predictions[0]
                
                self.analyzer.last_prediction = best_pred[0]
                self.analyzer.prediction_history.append(
                    (best_pred[0], text[-1], best_pred[1], datetime.now())
                )
                
                analysis = {
                    'history': list(self.analyzer.history)[-20:],
                    'streaks': self.analyzer.streaks,
                    'top_patterns': dict(sorted(self.analyzer.pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]),
                    'model_accuracies': self.ml_predictor.last_accuracies
                }
                
                response_lines = [
                    f"🔮 Prediction: {best_pred[0]} ({best_pred[1]*100:.1f}% confidence)",
                    f"📌 Method: {best_pred[2]}",
                    "",
                    f"📈 Recent: {''.join(analysis['history'][-30:])}",
                    f"🔥 Current Streaks: B={analysis['streaks']['B']} | S={analysis['streaks']['S']}",
                    f"🔄 Top Patterns:",
                ]
                
                for pattern, count in analysis['top_patterns'].items():
                    response_lines.append(f"- {pattern}: {count}x")
                
                if analysis['model_accuracies']:
                    response_lines.append("\n🤖 Model Accuracies:")
                    for model, acc in analysis['model_accuracies'].items():
                        response_lines.append(f"- {model.upper()}: {acc:.2f}")
                
                response = "\n".join(response_lines)
                
                if len(self.analyzer.history) > 10:
                    chart = self._generate_chart(analysis['history'])
                    if chart:
                        await update.message.reply_photo(photo=chart, caption=response)
                    else:
                        await update.message.reply_text(response)
                else:
                    await update.message.reply_text(response)
            else:
                # Show top 3 predictions below threshold when none meet the 90% confidence
                all_predictions.sort(key=lambda x: x[1], reverse=True)
                top_predictions = all_predictions[:3]
                
                response_lines = [
                    "⚠️ No predictions meeting 90% confidence threshold",
                    "",
                    "Top predictions below threshold:",
                ]
                
                for pred in top_predictions:
                    response_lines.append(
                        f"- {pred[0]} ({pred[1]*100:.1f}%): {pred[2]}"
                    )
                
                response_lines.extend([
                    "",
                    f"Current streaks: B={self.analyzer.streaks['B']} | S={self.analyzer.streaks['S']}",
                    f"Last 10: {''.join(list(self.analyzer.history)[-10:])}"
                ])
                
                await update.message.reply_text("\n".join(response_lines))
        def _generate_chart(self, history):
            if not history or len(history) < 2:
                return None
            
            plt.figure(figsize=(14, 6))
            try:
                y = [1 if x == 'B' else 0 for x in history]
                x = range(len(history))
                
                sns.set_style("whitegrid")
                plt.plot(x, y, 'o-', color='royalblue', linewidth=2, markersize=8, markerfacecolor='gold', markeredgewidth=1)
                plt.fill_between(x, y, color='royalblue', alpha=0.1)
                
                current_streak = self.analyzer.streaks['B'] or self.analyzer.streaks['S']
                if current_streak > 2 and current_streak <= len(history):
                    streak_start = len(history) - current_streak
                    plt.axvspan(streak_start, len(history), facecolor='orange', alpha=0.2)
                
                plt.title('Advanced Big/Small Trend Analysis', pad=20, fontsize=14)
                plt.yticks([0, 1], ['Small (S)', 'Big (B)'], fontsize=12)
                plt.xticks(fontsize=10)
                plt.xlabel('Last 20 Rounds', fontsize=12)
                
                mean_val = np.mean(y)
                plt.axhline(y=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                plt.legend(fontsize=10, loc='upper right')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                return InputFile(buf, filename='trend_analysis.png')
            except Exception as e:
                logging.error(f"Chart generation error: {str(e)}")
                plt.close()
                return None

    # --- Bot Setup ---
if __name__ == "__main__":
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            filename='wingo_bot.log'
        )
        
        bot = AdvancedWingoBot()
        app = Application.builder().token(TOKEN).build()
        
        app.add_handler(CommandHandler("start", bot.start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
        
        print("🚀 Advanced Wingo Bot with Enhanced Models and Stacking is running...")
        app.run_polling()