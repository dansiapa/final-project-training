import io, base64, pandas as pd, numpy as np, logging, shap, psutil, cv2, time, json, os
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
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor, XGBClassifier
import matplotlib
from scipy.stats import skew, kurtosis
from skimage.restoration import estimate_sigma

matplotlib.use('Agg')

# --- KONFIGURASI PERSISTENCE ---
CACHE_FILE = 'system_knowledge_cache.json'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- GLOBAL STATE (Intelligence Core) ---
state = {
    "models": {}, 
    "scaler": StandardScaler(), 
    "le_y": LabelEncoder(),
    "features": [], 
    "target": "", 
    "best_model_name": "", 
    "is_reg": True,
    "plot_fit": "",
    "plot_importance": "",
    "plot_shap": "",
    "image_cache": {},
    "training_data_forensic": []
}

# --- GLOBAL DOCUMENT RULES (Akurasi >80%) ---
DOCUMENT_RULES = {
    "KTP": {"min_sharp": 200, "max_ela": 3.0, "require_color": True},
    "SIM": {"min_sharp": 180, "max_ela": 3.5, "require_color": True},
    "PASPOR": {"min_sharp": 250, "max_ela": 2.5, "require_color": True},
    "NPWP": {"min_sharp": 150, "max_ela": 4.0, "require_color": True},
    "SLIP_GAJI": {"min_sharp": 120, "max_ela": 5.0, "require_color": False},
    "KARTU_KELUARGA": {"min_sharp": 150, "max_ela": 4.5, "require_color": False},
    "SURAT_DOMISILI": {"min_sharp": 130, "max_ela": 4.0, "require_color": False}
}

# --- PERSISTENCE LOGIC ---
def save_cache_to_disk():
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump({
                "image_cache": state["image_cache"],
                "training_data_forensic": state["training_data_forensic"]
            }, f)
    except Exception as e:
        logger.error(f"Save Cache Error: {e}")

def load_cache_from_disk():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                state["image_cache"] = data.get("image_cache", {})
                state["training_data_forensic"] = data.get("training_data_forensic", [])
            logger.info(f"📂 MEMORY LOADED: Berhasil memuat data dari disk.")
        except Exception as e:
            logger.error(f"❌ Gagal muat cache: {e}")

# --- HELPER: PLOT TO B64 ---
def get_b64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- ENGINE: FORENSIC ANALYSIS ---
class DeepForensicAuditor:
    @staticmethod
    def analyze(image_bytes):
        logger.info("Memulai proses Deep Forensic Audit (100 Tahap)...")
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Gagal mendecode gambar. Format file mungkin tidak didukung atau rusak.")
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            logger.info(f"Dimensi gambar terdeteksi: {w}x{h}")

            # --- [TAHAP 1-20] STATISTIK DISTRIBUSI PIKSEL ---
            pixel_stats = {
                "entropy": float(np.sum(-np.histogram(gray, 256, [0,256], density=True)[0] * 
                                np.log2(np.histogram(gray, 256, [0,256], density=True)[0] + 1e-7))),
                "skewness": float(skew(gray.flatten())),
                "kurtosis": float(kurtosis(gray.flatten())),
                "std_dev": float(np.std(gray))
            }
            logger.info(f"Statistik Piksel -> Entropy: {pixel_stats['entropy']:.2f}, Skewness: {pixel_stats['skewness']:.2f}")

            # --- [TAHAP 21-40] ERROR LEVEL ANALYSIS (ELA) & TAMPERING ---
            _, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resaved = cv2.imdecode(b, cv2.IMREAD_COLOR)
            ela_map = cv2.absdiff(img, resaved)
            ela_score = float(np.mean(ela_map))
            ela_inconsistency = float(np.std(ela_map))
            logger.info(f"Analisis ELA -> Score: {ela_score:.4f}, Inconsistency: {ela_inconsistency:.4f}")

            # --- [TAHAP 41-60] DOMAIN FREKUENSI (FFT) ---
            dft = np.fft.fft2(gray)
            fshift = np.fft.fftshift(dft)
            mag_spec = 20 * np.log(np.abs(fshift) + 1)
            fft_mean = float(np.mean(mag_spec))
            logger.info(f"Analisis Frekuensi (FFT) -> Mean Magnitude: {fft_mean:.2f}")

            # --- [TAHAP 61-80] ANALISIS TEKSTUR MIKRO (Grain & Noise) ---
            noise_sigma = float(estimate_sigma(gray, average_sigmas=True))
            grain_map = gray - cv2.GaussianBlur(gray, (0,0), 3)
            grain_energy = float(np.var(grain_map))
            logger.info(f"Analisis Tekstur -> Noise Sigma: {noise_sigma:.6f}, Grain Energy: {grain_energy:.2f}")

            # --- [TAHAP 81-100] TIPOGRAFI, CAHAYA & GEOMETRI ---
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(np.sum(edges > 0) / (h * w))
            
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            lighting_grad = float(np.std(cv2.magnitude(gx, gy)))
            
            aspect_ratio = float(w / h)
            is_color = not (np.allclose(img[:,:,0], img[:,:,1], atol=2))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            logger.info(f"Analisis Fisik -> Sharpness: {sharpness:.2f}, Edge Density: {edge_density:.4f}, Color: {is_color}")

            logger.info("Deep Forensic Audit selesai dengan sukses.")
            
            return {
                "stats": pixel_stats, "ela": ela_score, "ela_std": ela_inconsistency,
                "fft": fft_mean, "noise": noise_sigma, "grain": grain_energy,
                "edge": edge_density, "lighting": lighting_grad, "ratio": aspect_ratio,
                "is_color": is_color, "sharp": sharpness
            }

        except Exception as e:
            logger.error(f"Terjadi kesalahan fatal selama Deep Forensic Analysis: {str(e)}", exc_info=True)
            return None

# --- ROUTES: WEB & HEALTH ---
@app.route('/')
def index(): return render_template('login.html')

@app.route('/home')
def home(): return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "Online",
        "memory_usage": f"{psutil.virtual_memory().percent}%",
        "cached_docs": len(state["image_cache"]),
        "knowledge_base": len(state["training_data_forensic"]),
        "cpu_usage": f"{psutil.cpu_percent()}%"
    })

# --- ROUTES: DOCUMENT INTELLIGENCE ---
@app.route('/teach_document', methods=['POST'])
def teach_document():
    file = request.files['file']
    category = request.form.get('category').upper()
    is_authentic = int(request.form.get('is_authentic', 1))
    
    # Validasi kategori dari list dokumen yang sudah di-define
    all_valid = [item for sub in state["doc_categories"].values() for item in sub]
    if category not in all_valid:
        return jsonify({"error": f"Kategori {category} tidak dikenal sistem"}), 400

    features, _ = enhanced_forensic_analysis(file.read(), file.filename, force_update=True)
    if features:
        state["training_data_forensic"].append({**features, "category": category, "label": is_authentic})
        save_cache_to_disk()
        return jsonify({"status": "success", "msg": f"Sistem berhasil mempelajari {category}"})
    return jsonify({"error": "Analisis gagal"}), 500

import cv2
import numpy as np
import base64

@app.route('/verify_document', methods=['POST'])
def verify_document():
    if 'file' not in request.files: 
        logger.warning("Request tanpa file")
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    category = request.form.get('category', 'KTP').upper()
    rules = DOCUMENT_RULES.get(category, DOCUMENT_RULES.get('KTP'))
    
    logger.info(f"Audit dimulai: {category}")
    
    content = file.read()
    phys = DeepForensicAuditor.analyze(content)
    
    if not phys: 
        logger.error(f"Analisis gagal: {category}")
        return jsonify({"error": "Analisis gagal"}), 400

    score = 100
    violations = []

    if phys['fft'] > 165: 
        score -= 65 
        violations.append("CRITICAL: Media Integrity - Screen Reshoot")
        logger.info(f"[{category}] FFT High: {phys['fft']:.2f}")

    if phys['ela_std'] > 12: 
        score -= 40
        violations.append("CRITICAL: Pixel Integrity - Digital Tampering")
        logger.info(f"[{category}] ELA STD High: {phys['ela_std']:.2f}")

    if phys['grain'] < 0.5:
        score -= 20
        violations.append("Texture Integrity: Digital Design")
        logger.info(f"[{category}] Grain Low: {phys['grain']:.4f}")

    if phys['lighting'] < 8:
        score -= 15
        violations.append("Lighting Integrity: Flat Lighting")
        logger.info(f"[{category}] Lighting Low: {phys['lighting']:.2f}")

    min_sharp = rules.get('min_sharp', 500)
    if phys['sharp'] < min_sharp:
        score -= 10
        violations.append(f"Visual Quality: Blurry")
        logger.info(f"[{category}] Sharpness Low: {phys['sharp']:.2f}")

    final_score = max(0, score)
    
    if final_score >= 90:
        verdict = "VALID (ASLI)"
    elif final_score >= 75:
        verdict = "WARNING (REVISI)"
    else:
        verdict = "INVALID (PALSU/MANIPULASI)"

    logger.info(f"Audit Selesai [{category}] -> Skor: {final_score}, Verdict: {verdict}")

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "success": True,
        "category": category,
        "trust_score": f"{round(final_score, 2)}%",
        "verdict": verdict,
        "anomalies_detected": violations,
        "forensic_log": phys,
        "processed_image": img_base64
    })

@app.route('/process_cv', methods=['POST'])
def process_cv():
    file = request.files['image']
    category = request.form.get('mode', 'KTP').upper() # Mode reuse sebagai category
    rules = DOCUMENT_RULES.get(category, DOCUMENT_RULES['KTP'])
    
    content = file.read()
    phys = DeepForensicAuditor.analyze(content)
    
    score = 0
    flags = []
    if phys['fft'] < 165: score += 40 
    else: flags.append("Screen Reshoot Detected")
    
    if phys['ela'] < rules['max_ela']: score += 30
    else: flags.append("Digital Tampering Detected")
    
    if phys['sharp'] > rules['min_sharp']: score += 30
    else: flags.append("Blurry/Low Quality")

    # Encode Image for Preview
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    _, buffer = cv2.imencode('.png', img)
    
    return jsonify({
        "success": True,
        "processed_image": base64.b64encode(buffer).decode('utf-8'),
        "label": "ASLI" if score >= 80 else "MENCURIGAKAN",
        "confidence": score,
        "flags": flags
    })

# --- ROUTES: AUTO-ML ENGINE (FULL ORIGINAL LOGIC) ---
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
            "K-Nearest Neighbors": KNeighborsRegressor() if is_reg else KNeighborsClassifier(),
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
    load_cache_from_disk()
    logger.info("🚀 SISTEM INTELEGENSI DOKUMEN & AUTO-ML AKTIF.")
    app.run(debug=False, host='0.0.0.0', port=8080)