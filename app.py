import io, base64, pandas as pd, numpy as np, matplotlib, psutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
import psutil
import GPUtil

app = Flask(__name__)
CORS(app)

state = {
    "models": {}, "scaler": StandardScaler(), "le_y": LabelEncoder(),
    "features": [], "is_reg": True, "target": "", "df_processed": None,
    "best_model_name": "", "categorical_maps": {}, 
    "numeric_defaults": {}, "original_cols": []
}

def get_b64():
    try:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close('all')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except: return ""

@app.route('/')
def index(): return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    gpus = GPUtil.getGPUs()
    gpu_stats = []
    
    for gpu in gpus:
        gpu_stats.append({
            "id": gpu.id,
            "name": gpu.name,
            "load": f"{gpu.load * 100:.1f}%",
            "memory_used": f"{gpu.memoryUsed}MB",
            "memory_total": f"{gpu.memoryTotal}MB",
            "temperature": f"{gpu.temperature} °C"
        })

    return jsonify({
        "status": "Online",
        "memory_usage": f"{psutil.virtual_memory().percent}%",
        "gpu_usage": gpu_stats if gpu_stats else "No GPU detected",
        "engine": "Unlimited Chunking"
    })

@app.route('/get_columns', methods=['POST'])
def get_columns():
    try:
        df = pd.read_csv(request.files['file'], nrows=5)
        return jsonify({"columns": df.columns.tolist()})
    except Exception as e: return jsonify({"error": str(e)}), 400

@app.route('/train', methods=['POST'])
def train():
    global state
    try:
        file = request.files['file']
        target = request.form['target']
        
        preview = pd.read_csv(file, nrows=100).dropna()
        file.seek(0)
        is_reg = not (preview[target].dtype == 'object' or preview[target].nunique() < 10)
        state.update({"target": target, "is_reg": is_reg, "original_cols": [c for c in preview.columns if c != target]})

        chunk_list = []
        y_list = []
        first_chunk = True
        total_rows_processed = 0
        chunk_count = 0
        
        print("\n--- START PROCESSING ALL DATA ---")
        
        for chunk in pd.read_csv(file, chunksize=10000):
            chunk = chunk.dropna()
            if chunk.empty: continue
            
            chunk_count += 1
            total_rows_processed += len(chunk)
            
            print(f"Reading Chunk #{chunk_count}: +{len(chunk)} rows (Current Total: {total_rows_processed})")
            
            X_batch = pd.get_dummies(chunk.drop(columns=[target]))
            y_batch = chunk[target]
            
            if first_chunk:
                state["features"] = list(X_batch.columns)
                state["numeric_defaults"] = chunk.select_dtypes(include=[np.number]).median().to_dict()
                state["categorical_maps"] = {col: chunk[col].unique().tolist() for col in chunk.select_dtypes(include=['object']).columns}
                first_chunk = False
            
            X_batch = X_batch.reindex(columns=state["features"], fill_value=0)
            
            chunk_list.append(X_batch)
            y_list.append(y_batch)

        print(f"--- FINISHED READING: {total_rows_processed} rows loaded ---")

        X_final = pd.concat(chunk_list)
        y_final = pd.concat(y_list)
        
        del chunk_list 
        del y_list

        if not is_reg:
            y_final = state["le_y"].fit_transform(y_final)
            
        X_sc = state["scaler"].fit_transform(X_final)
        state["df_processed"] = pd.DataFrame(X_sc, columns=state["features"])
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_final, test_size=0.2, random_state=42)

        models_config = {
            "XGBoost": XGBRegressor(n_estimators=100, n_jobs=1, random_state=42) if is_reg 
                       else XGBClassifier(n_estimators=100, n_jobs=1, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42) if is_reg 
                             else RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42),
            "Linear/Log Reg": Ridge() if is_reg 
                              else LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeRegressor(random_state=42) if is_reg 
                             else DecisionTreeClassifier(random_state=42)
        }

        metrics = []
        best_score = -999
        for name, m in models_config.items():
            print(f"Training {name}...")
            m.fit(X_tr, y_tr)
            score = r2_score(y_te, m.predict(X_te)) if is_reg else accuracy_score(y_te, m.predict(X_te))
            state["models"][name] = m
            metrics.append({"name": name, "accuracy": score})
            if score > best_score:
                best_score = score
                state["best_model_name"] = name

        print(f"Training Complete. Best: {state['best_model_name']} ({best_score})\n")

        return jsonify({
            "metrics": metrics, "best_model": state["best_model_name"], "target": target,
            "defaults": state["numeric_defaults"], "categories": state["categorical_maps"], 
            "original_cols": state["original_cols"]
        })
    except Exception as e: 
        print(f"CRITICAL ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_analytics', methods=['GET'])
def get_analytics():
    try:
        import shap
        m = state["models"][state["best_model_name"]]
        X_test = state["df_processed"].head(10)
        X_bg = state["df_processed"].sample(n=min(15, len(state["df_processed"])))
        plots = {"fit": "", "shap": "", "importance": ""}
        plt.figure(figsize=(6,4))
        plt.plot(m.predict(X_test.values), 'o-', color='#0d6efd')
        plots['fit'] = get_b64()
        explainer = shap.Explainer(m.predict, X_bg)
        shap_values = explainer(X_test)
        plt.figure(figsize=(6,4))
        shap.plots.bar(shap_values, max_display=7, show=False)
        plots['importance'] = get_b64()
        plt.figure(figsize=(8,4))
        shap.summary_plot(shap_values, X_test, show=False)
        plots['shap'] = get_b64()
        return jsonify({"plots": plots})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        import shap
        m = state["models"][state["best_model_name"]]
        raw_input = request.json['inputs']
        input_df = pd.DataFrame([raw_input])
        input_encoded = pd.get_dummies(input_df).reindex(columns=state["features"], fill_value=0)
        input_sc = state["scaler"].transform(input_encoded)
        res = m.predict(input_sc)[0]
        X_bg = state["df_processed"].sample(n=min(10, len(state["df_processed"])))
        explainer = shap.Explainer(m.predict, X_bg)
        sv = explainer(input_sc)
        plt.figure(figsize=(10,5))
        exp = shap.Explanation(values=sv.values[0], base_values=sv.base_values[0], data=input_sc[0], feature_names=state["features"])
        shap.plots.waterfall(exp, show=False)
        w_img = get_b64()
        if not state["is_reg"]:
            res = state["le_y"].inverse_transform([int(res)])[0]
        return jsonify({"prediction": str(res), "waterfall": w_img})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=False)