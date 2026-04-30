import io, base64, pandas as pd, numpy as np, logging, shap, psutil
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib
import time

matplotlib.use('Agg')

# --- KONFIGURASI LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global State - Menyimpan metadata dan model untuk siklus hidup aplikasi
state = {
    "models": {}, 
    "scaler": StandardScaler(), 
    "le_y": LabelEncoder(),
    "features": [], 
    "original_features": [], 
    "target": "", 
    "df_raw": None, 
    "X_test": None,
    "y_test": None,
    "best_model_name": "", 
    "categorical_maps": {}, 
    "numeric_defaults": {},
    "is_reg": True,
    "model_metrics": {} 
}

def get_b64():
    """Fungsi pembantu untuk mengubah plot matplotlib menjadi base64 string."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index(): 
    return render_template('login.html')

@app.route('/home')
def home(): 
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "Online",
        "memory_usage": f"{psutil.virtual_memory().percent}%",
        "gpu_usage": "N/A"
    })

@app.route('/get_columns', methods=['POST'])
def get_cols():
    file = request.files['file']
    df = pd.read_csv(file)
    df.columns = [col.strip().replace('_', ' ').upper() for col in df.columns]
    return jsonify({"columns": df.columns.tolist()})

@app.route('/train', methods=['POST'])
def train():
    global state
    try:
        file = request.files['file']
        target_input = request.form['target'].strip()
        
        logger.info(f"=== [START] PROSES TRAINING DIMULAI (Target Awal: {target_input}) ===")

        # 1. LOAD DATA DENGAN CHUNKING & HEADER FORMATTING
        file.seek(0)
        chunk_list = []
        chunk_idx = 0
        logger.info("Memulai pembacaan file CSV dengan sistem chunking & header formatting...")
        
        for chunk in pd.read_csv(file, chunksize=5000):
            chunk_idx += 1
            chunk.columns = [col.strip().replace('_', ' ').upper() for col in chunk.columns]
            chunk = chunk.dropna()
            chunk_list.append(chunk)
            
        df = pd.concat(chunk_list, ignore_index=True)
        target = target_input.replace('_', ' ').upper()
        del chunk_list 
        
        logger.info(f"Dataframe siap. Total baris: {len(df)}, Total kolom: {len(df.columns)}")

        # 2. DYNAMIC BIAS GUARD (AUDIT AGRESIF)
        logger.info("Menjalankan 'Dynamic Bias Guard' Agresif (Ambang Batas > 15%)...")
        X_audit = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
        y_audit = pd.factorize(df[target])[0]
        
        is_reg = not (df[target].dtype == 'object' or df[target].nunique() < 10)
        logger.info(f"Deteksi tipe problem: {'Regression' if is_reg else 'Classification'}")

        auditor = RandomForestRegressor(n_estimators=50, max_depth=5) if is_reg else \
                  RandomForestClassifier(n_estimators=50, max_depth=5)
        
        auditor.fit(X_audit, y_audit)
        importances = pd.Series(auditor.feature_importances_, index=X_audit.columns)
        
        features_to_drop = set()
        leaks = importances[importances > 0.15].sort_values(ascending=False).index.tolist()
        
        for l in leaks:
            orig_col = next((c for c in df.columns if l.startswith(c)), l)
            if orig_col != target:
                logger.warning(f"!!! BIAS TERDETEKSI !!! Kolom '{orig_col}' mendominasi {importances[l]*100:.2f}%")
                features_to_drop.add(orig_col)

        removed_cols = list(features_to_drop)
        if removed_cols:
            logger.warning(f"MENGHAPUS KOLOM BIAS SECARA OTOMATIS: {removed_cols}")
            df = df.drop(columns=removed_cols)
        else:
            logger.info("Audit selesai: Tidak ditemukan kolom dengan dominasi ekstrem.")

        # 3. PREPROCESSING & METADATA UI
        logger.info("Mengekstrak metadata untuk sinkronisasi UI (Auto-fill & Dropdowns)...")
        state["df_raw"] = df
        state["target"] = target
        state["is_reg"] = is_reg
        state["original_features"] = df.drop(columns=[target]).columns.tolist()
        state["numeric_defaults"] = df.select_dtypes(include=[np.number]).median().to_dict()
        state["categorical_maps"] = {
            col: df[col].unique().tolist() 
            for col in df.select_dtypes(include=['object', 'string']).columns 
            if col != target
        }

        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target] if is_reg else state["le_y"].fit_transform(df[target])
        state["features"] = X.columns.tolist()
            
        X_sc = state["scaler"].fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
        state["X_test"], state["y_test"] = X_te, y_te
        logger.info("Preprocessing selesai. Split data 80:20 berhasil.")

        # 4. OMNI-MODEL COMPETITION DENGAN METRIK LENGKAP
        logger.info("Memulai Kompetisi Model (Omni-Model Competition)...")
        models_pool = {
            "XGBoost": XGBRegressor(n_estimators=100) if is_reg else XGBClassifier(n_estimators=100),
            "Random Forest": RandomForestRegressor(n_estimators=100) if is_reg else RandomForestClassifier(n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100) if is_reg else GradientBoostingClassifier(n_estimators=100),
            "Decision Tree": DecisionTreeRegressor() if is_reg else DecisionTreeClassifier(),
            "Linear/Log Reg": Ridge() if is_reg else LogisticRegression(max_iter=1000)
        }

        summary_metrics = []
        best_score = -float('inf')
        
        for name, m in models_pool.items():
            logger.info(f"Training Engine: {name}...")
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            
            res = {"name": name}
            if is_reg:
                res["accuracy"] = max(0, r2_score(y_te, preds))
                res["mae"] = mean_absolute_error(y_te, preds)
                res["mape"] = mean_absolute_percentage_error(y_te, preds)
                res["f1"] = 0 
                logger.info(f"Hasil {name} - R2: {res['accuracy']:.4f}, MAE: {res['mae']:.4f}")
            else:
                res["accuracy"] = accuracy_score(y_te, preds)
                res["f1"] = f1_score(y_te, preds, average='weighted')
                res["mae"], res["mape"] = 0, 0
                logger.info(f"Hasil {name} - Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}")
            
            state["models"][name] = m
            state["model_metrics"][name] = res
            summary_metrics.append(res)
            
            if res["accuracy"] > best_score:
                best_score = res["accuracy"]
                state["best_model_name"] = name

        logger.info(f"🏆 PEMENANG: {state['best_model_name']} (Score: {best_score:.4f})")
        logger.info("=== [FINISH] SELURUH PROSES TRAINING SELESAI ===")

        return jsonify({
            "status": "success",
            "best_model": state["best_model_name"],
            "metrics": summary_metrics,
            "removed_features": removed_cols,
            "original_cols": state["original_features"], 
            "defaults": state["numeric_defaults"],       
            "categories": state["categorical_maps"]      
        })

    except Exception as e:
        logger.error(f"!!! FATAL ERROR DI /TRAIN !!!: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get_analytics')
def get_analytics():
    if state["df_raw"] is None: 
        logger.error("Gagal memuat analytics: Data kosong.")
        return jsonify({"error": "No data"}), 400
    
    logger.info("=== [START] PROSES GENERASI ANALYTICS ===")
    plots = {}
    model_name = state["best_model_name"]
    model = state["models"][model_name]
    y_te, X_te = state["y_test"], state["X_test"]
    preds = model.predict(X_te)

    try:
        # 1. Performance Comparison Chart
        logger.info("Generating Model Performance Comparison Chart...")
        df_m = pd.DataFrame(state["model_metrics"].values())
        plt.figure(figsize=(10, 5))
        if state["is_reg"]:
            df_plot = df_m.melt(id_vars="name", value_vars=["accuracy", "mae"])
            sns.barplot(data=df_plot, x="name", y="value", hue="variable")
            plt.title("PERFORMANCE: ACCURACY (R2) VS MAE")
        else:
            df_plot = df_m.melt(id_vars="name", value_vars=["accuracy", "f1"])
            sns.barplot(data=df_plot, x="name", y="value", hue="variable")
            plt.title("PERFORMANCE: ACCURACY VS F1-SCORE")
        plots["fit"] = get_b64()

        # 2. Error Distribution Chart
        logger.info("Generating Error Distribution Chart...")
        plt.figure(figsize=(8, 5))
        if state["is_reg"]:
            errors = y_te - preds
            sns.histplot(errors, kde=True, color='red')
            plt.title("ERROR DISTRIBUTION (RESIDUALS)")
            plt.xlabel("Difference (Actual - Predicted)")
        else:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_te, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("ERROR DISTRIBUTION (CONFUSION MATRIX)")
        plots["importance"] = get_b64()

        # 3. SHAP Summary Plot
        logger.info("Memulai perhitungan SHAP Summary...")
        try:
            plt.figure()
            explainer = shap.Explainer(model, X_te)
            shap_values = explainer(X_te[:100])
            clean_features = [f.replace('_', ' ').upper() for f in state["features"]]
            shap.summary_plot(shap_values, X_te[:100], feature_names=clean_features, show=False)
            plots["shap"] = get_b64()
            logger.info("SHAP Plot berhasil dibuat.")
        except Exception as shap_err: 
            logger.warning(f"Gagal generate SHAP: {shap_err}")
            plots["shap"] = None

        logger.info("=== [FINISH] SEMUA PLOT ANALYTICS BERHASIL DIKIRIM ===")
        return jsonify({"plots": plots})
    except Exception as e:
        logger.error(f"Error Analytics: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("=== [START] PROSES PREDIKSI INDIVIDUAL ===")
    try:
        raw_input = request.json.get('inputs', {})
        processed_input = {k.upper().replace('_', ' '): v for k, v in raw_input.items()}
        logger.info(f"Input diterima: {processed_input}")
        
        input_df = pd.DataFrame([processed_input])
        input_enc = pd.get_dummies(input_df).reindex(columns=state["features"], fill_value=0)
        input_sc = state["scaler"].transform(input_enc)
        
        model = state["models"][state["best_model_name"]]
        pred = model.predict(input_sc)[0]
        
        if not state["is_reg"]: 
            pred = state["le_y"].inverse_transform([int(pred)])[0]
        
        logger.info(f"Hasil Prediksi: {pred}")

        # SHAP Waterfall dengan Agregasi ke Kolom Asli
        waterfall_b64 = None
        logger.info("Generating Waterfall Plot (Aggregated)...")
        try:
            explainer = shap.Explainer(model, state["X_test"])
            shap_values = explainer(input_sc)
            orig_cols = state["original_features"]
            agg_shap = [shap_values.values[0, [j for j, f in enumerate(state["features"]) if f.startswith(col)]].sum() for col in orig_cols]

            new_exp = shap.Explanation(
                values=np.array(agg_shap), 
                base_values=shap_values.base_values[0], 
                data=input_df.values[0], 
                feature_names=orig_cols
            )

            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(new_exp, show=False)
            waterfall_b64 = get_b64()
            logger.info("Waterfall Plot siap.")
        except Exception as shap_err: 
            logger.warning(f"Waterfall Plot skip: {shap_err}")

        logger.info("=== [FINISH] PREDIKSI SELESAI ===")
        return jsonify({
            "prediction": str(round(pred, 2) if state["is_reg"] else pred), 
            "waterfall": waterfall_b64
        })
    except Exception as e: 
        logger.error(f"Error Predict: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)