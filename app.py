# app.py
from flask import Flask, request, render_template_string
import os, re, difflib, joblib, math
import pandas as pd

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ----------------- HTML TEMPLATE (keeps form values after POST) -----------------
html_page = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Skill → Role & Salary Prediction</title>
  <style>
    body { font-family: 'Poppins',sans-serif; background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:#fff; margin:0; padding:30px; }
    .container { max-width:900px; margin:0 auto; background: rgba(255,255,255,0.06); padding:28px; border-radius:14px; border:1px solid rgba(255,255,255,0.08); }
    h2 { text-align:center; color:#ffdd55; margin:0 0 18px; }
    .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-bottom:8px; }
    input[type=text], select { width:100%; padding:8px 10px; border-radius:8px; border:none; background: rgba(255,255,255,0.9); color:#000; font-size:14px; }
    option { color:#000 !important; background:#fff !important; }
    label { font-size:13px; color:#eee; display:block; margin-bottom:6px; }
    button { width:100%; padding:12px; border-radius:30px; border:none; background:#ffdd33; color:#222; font-weight:700; cursor:pointer; margin-top:14px; }
    .result { margin-top:20px; padding:18px; border-radius:10px; background: rgba(0,0,0,0.35); border-left:6px solid #ffdd33; }
    .role { font-size:20px; color:#fff; margin:6px 0; }
    .salary { font-size:18px; color:#bfffbf; margin:4px 0; }
    .small { font-size:13px; color:#ddd; }
    .warn { color:#ffd699; margin-top:8px; }
    .error { color:#ff8080; margin-top:8px; }
    .suggest { color:#cfe8ff; margin-top:8px; }
    .top-list { margin-top:8px; }
    .top-item { background: rgba(255,255,255,0.03); padding:8px; border-radius:6px; margin-bottom:6px; }
    .meta { font-size:13px; color:#ddd; margin-top:6px; }
  </style>
</head>
<body>
  <div class="container">
    <h2>Skill → Role & Salary Prediction</h2>

    <form method="POST">
      <div class="grid">
        <div>
          <label>Skill 1</label>
          <input type="text" name="skill1" placeholder="e.g., Python" value="{{ form_values.skill1|default('') }}">
        </div>
        <div>
          <label>Skill 2</label>
          <input type="text" name="skill2" placeholder="e.g., SQL" value="{{ form_values.skill2|default('') }}">
        </div>
        <div>
          <label>Skill 3</label>
          <input type="text" name="skill3" placeholder="e.g., Power BI" value="{{ form_values.skill3|default('') }}">
        </div>
        <div>
          <label>Skill 4</label>
          <input type="text" name="skill4" placeholder="e.g., Machine Learning" value="{{ form_values.skill4|default('') }}">
        </div>
        <div>
          <label>Skill 5</label>
          <input type="text" name="skill5" placeholder="e.g., Docker" value="{{ form_values.skill5|default('') }}">
        </div>
        <div>
          <label>Skill 6 (optional)</label>
          <input type="text" name="skill6" placeholder="e.g., Communication" value="{{ form_values.skill6|default('') }}">
        </div>
      </div>

      <div class="grid" style="margin-top:10px;">
        <div>
          <label>Experience level</label>
          <select name="experience">
            <option value="fresher" {% if form_values.experience=='fresher' %}selected{% endif %}>Fresher</option>
            <option value="1-3" {% if form_values.experience=='1-3' %}selected{% endif %}>1 - 3 years</option>
            <option value="3-5" {% if form_values.experience=='3-5' %}selected{% endif %}>3 - 5 years</option>
            <option value="5+" {% if form_values.experience=='5+' %}selected{% endif %}>5+ years</option>
          </select>
        </div>
        <div>
          <label>City</label>
          <select name="city">
            <option value="remote" {% if form_values.city=='remote' %}selected{% endif %}>Remote</option>
            <option value="bangalore" {% if form_values.city=='bangalore' %}selected{% endif %}>Bangalore</option>
            <option value="pune" {% if form_values.city=='pune' %}selected{% endif %}>Pune</option>
            <option value="hyderabad" {% if form_values.city=='hyderabad' %}selected{% endif %}>Hyderabad</option>
            <option value="delhi" {% if form_values.city=='delhi' %}selected{% endif %}>Delhi</option>
            <option value="other" {% if form_values.city=='other' %}selected{% endif %}>Other</option>
          </select>
        </div>
      </div>

      <div class="row">
        <button type="submit">Predict Role & Salary</button>
      </div>
    </form>

    {% if message %}
      <div class="result">
        <div class="{{ message_class }}">{{ message }}</div>
        {% if suggestions %}
          <div class="suggest"><strong>Suggestions:</strong> {{ suggestions }}</div>
        {% endif %}
      </div>
    {% endif %}

    {% if role %}
      <div class="result">
        <div class="role"><strong>Recommended Role:</strong> {{ role }}</div>
        <div class="salary"><strong>Estimated Salary:</strong> {{ salary }}</div>
        <div class="meta">Based on skills: {{ used_skills }}</div>

        <div class="top-list">
          <div class="small">Top 3 role predictions:</div>
          {% for r, p in top3 %}
            <div class="top-item"><b>{{ r }}</b> — {{ "%.1f"|format(p*100) }}% probability</div>
          {% endfor %}
        </div>

        {% if warning_text %}
          <div class="warn">{{ warning_text }}</div>
        {% endif %}
      </div>
    {% endif %}

  </div>
</body>
</html>
"""

# ----------------- PATHS & MODEL LOADING -----------------
MODEL_PATHS = ["models", "/mnt/data/models", os.path.join(os.getcwd(), "models")]
loaded = False
role_pipe = None
salary_pipe = None
label_encoder = None

for mp in MODEL_PATHS:
    try:
        role_pipe = joblib.load(os.path.join(mp, "role_pipeline.joblib"))
        salary_pipe = joblib.load(os.path.join(mp, "salary_pipeline.joblib"))
        label_encoder = joblib.load(os.path.join(mp, "label_encoder_role.joblib"))
        loaded = True
        print("Loaded models from:", mp)
        break
    except Exception:
        continue

if not loaded:
    print("Models not found in paths, the app will use fallback rule-based predictor.")

# ----------------- Build skill vocabulary from CSV or fallback -----------------
CSV_CANDIDATES = ["india_job_dataset_50000.csv", "/mnt/data/india_job_dataset_50000.csv"]

def build_skill_vocabulary():
    for c in CSV_CANDIDATES:
        if os.path.exists(c):
            try:
                df = pd.read_csv(c)
                skills_col = df.get("skills")
                if skills_col is not None:
                    sset = set()
                    for s in skills_col.dropna().astype(str):
                        parts = re.split(r'[;,/]|,', s)
                        for p in parts:
                            t = p.strip().lower()
                            if t:
                                sset.add(t)
                    return sset
            except Exception:
                pass
    return set([
        "python","sql","machine learning","pandas","scikit-learn","excel","power bi",
        "tensorflow","pytorch","java","spring boot","apis","javascript","react","html","css",
        "node.js","mongodb","docker","kubernetes","aws","azure","terraform","linux",
        "siem","network security","ethical hacking","seo","google ads","analytics",
        "deep learning","nlp","pandas","powerbi"
    ])

ALL_SKILLS = build_skill_vocabulary()

# ----------------- Helpers -----------------
def tokenize_input(text):
    t = text.lower()
    t = re.sub(r'[^a-z0-9\s\.\-]', ' ', t)
    tokens = re.split(r'\s+|[,;\/\|]+', t)
    tokens = [tok.strip() for tok in tokens if tok.strip()]
    return tokens

def looks_like_noise(token):
    if re.fullmatch(r'\d+', token): return True
    if re.fullmatch(r'[-_]+', token): return True
    letters = re.findall(r'[a-z]', token)
    if len(letters) < 2 and any(c.isdigit() for c in token): return True
    return False

def fuzzy_correct(token):
    matches = difflib.get_close_matches(token, ALL_SKILLS, n=1, cutoff=0.6)
    return matches[0] if matches else None

def validate_and_autocorrect(skills_list):
    joined = " ".join(skills_list)
    tokens = tokenize_input(joined)
    used = []
    invalid = []
    suggestions = {}
    auto_fixed = {}
    for tok in tokens:
        if looks_like_noise(tok):
            invalid.append(tok)
            continue
        if tok in ALL_SKILLS:
            used.append(tok)
            continue
        substring_matches = [s for s in ALL_SKILLS if tok in s]
        if substring_matches:
            used.append(min(substring_matches, key=len))
            continue
        corr = fuzzy_correct(tok)
        if corr:
            auto_fixed[tok] = corr
            used.append(corr)
            suggestions[tok] = [corr]
        else:
            invalid.append(tok)
    used = list(dict.fromkeys(used))
    return used, invalid, suggestions, auto_fixed

def format_salary_numeric_to_lpa_range(sal_numeric, exp_mult=1.0, city_mult=1.0):
    adjusted = sal_numeric * exp_mult * city_mult
    low = int(math.floor(adjusted * 0.85))
    high = int(math.ceil(adjusted * 1.15))
    low_lpa = low / 1e5
    high_lpa = high / 1e5
    def fmt(x):
        if x == int(x):
            return f"{int(x)}"
        return f"{x:.1f}"
    return f"₹{fmt(low_lpa)} - {fmt(high_lpa)} LPA (est)"

EXP_MULT = {"fresher": 0.7, "1-3": 0.95, "3-5": 1.15, "5+": 1.35}
CITY_MULT = {"remote":1.00,"bangalore":1.25,"pune":1.10,"hyderabad":1.10,"delhi":1.15,"other":1.00}

def predict_with_models(input_text, exp_bucket, city_key):
    top3 = []
    if loaded and role_pipe is not None and salary_pipe is not None and label_encoder is not None:
        try:
            probs = None
            if hasattr(role_pipe, "predict_proba"):
                probs = role_pipe.predict_proba([input_text])[0]
                idxs = probs.argsort()[::-1][:3]
                top3 = [(label_encoder.inverse_transform([i])[0], float(probs[i])) for i in idxs]
            else:
                pred_idx = role_pipe.predict([input_text])[0]
                top3 = [(label_encoder.inverse_transform([pred_idx])[0], 1.0)]
        except Exception:
            top3 = []
        role_name = top3[0][0] if top3 else "Unknown Role"
        try:
            salary_pred = salary_pipe.predict([input_text])[0]
            salary_num = int(round(float(salary_pred)))
        except Exception:
            salary_num = None
    else:
        role_name, sal_text = fallback_predict_simple(input_text)
        top3 = [(role_name, 0.9)]
        salary_num = None
        return role_name, sal_text, top3, salary_num

    exp_mult = EXP_MULT.get(exp_bucket, 1.0)
    city_mult = CITY_MULT.get(city_key, 1.0)
    if salary_num is not None:
        salary_str = format_salary_numeric_to_lpa_range(salary_num, exp_mult=exp_mult, city_mult=city_mult)
    else:
        salary_str = "₹2 - 5 LPA (est)"
    return role_name, salary_str, top3, salary_num

def fallback_predict_simple(s):
    s = s.lower()
    if ("python" in s and ("ml" in s or "machine learning" in s or "deep learning" in s)):
        return "Data Scientist", "₹6 - 12 LPA"
    if ("sql" in s and ("excel" in s or "power bi" in s or "powerbi" in s)):
        return "Data Analyst", "₹3 - 7 LPA"
    if ("javascript" in s or "react" in s or "html" in s or "css" in s):
        return "Frontend Developer", "₹4 - 8 LPA"
    if ("java" in s or "spring" in s or "spring boot" in s):
        return "Backend Developer", "₹4 - 9 LPA"
    if ("aws" in s or "kubernetes" in s or "docker" in s or "terraform" in s):
        return "DevOps / Cloud Engineer", "₹6 - 14 LPA"
    if ("ethical hacking" in s or "siem" in s or "network security" in s):
        return "Security Analyst / Penetration Tester", "₹5 - 12 LPA"
    return "General IT Role", "₹2 - 5 LPA"

# ----------------- Flask route (preserve form values) -----------------
@app.route("/", methods=["GET", "POST"])
def home():
    role = None
    salary = None
    top3 = []
    message = None
    message_class = "error"
    suggestions_text = None
    warning_text = None
    used_skills = ""
    # default form values (so template can render them always)
    form_values = {
        "skill1": "", "skill2": "", "skill3": "", "skill4": "", "skill5": "", "skill6": "",
        "experience": "fresher", "city": "remote"
    }

    if request.method == "POST":
        # collect raw inputs and store them into form_values so they persist
        for key in ["skill1","skill2","skill3","skill4","skill5","skill6"]:
            form_values[key] = request.form.get(key,"").strip()
        form_values["experience"] = request.form.get("experience","fresher")
        form_values["city"] = request.form.get("city","remote")

        skills = [form_values[k] for k in ["skill1","skill2","skill3","skill4","skill5","skill6"] if form_values[k]]
        experience = form_values["experience"]
        city = form_values["city"]

        if not skills:
            message = "Enter skill"
            message_class = "error"
            return render_template_string(html_page, form_values=form_values, message=message, message_class=message_class)

        used, invalid, suggestions, auto_fixed = validate_and_autocorrect(skills)
        used_skills = ", ".join(used) if used else ""

        if not used:
            if suggestions:
                suggestions_text = " ; ".join([f"{k} → {','.join(v)}" for k,v in suggestions.items()])
                message = "Enter proper spelling"
                message_class = "error"
            else:
                message = "Enter skill"
                message_class = "error"
            return render_template_string(html_page, form_values=form_values, message=message, message_class=message_class, suggestions=suggestions_text)

        input_text = " ".join(used)
        role_name, salary_str, top3_list, salary_num = predict_with_models(input_text, experience, city)

        if invalid:
            if suggestions:
                suggestions_text = " ; ".join([f"{k} → {','.join(v)}" for k,v in suggestions.items()])
                warning_text = "Some inputs were not recognized. Suggestions: " + suggestions_text
            else:
                warning_text = "Some inputs were not recognized and were ignored."

        if auto_fixed and not suggestions_text:
            suggestions_text = " ; ".join([f"{k} → {v}" for k,v in auto_fixed.items()])

        role = role_name
        salary = salary_str
        top3 = top3_list

    # on GET or after processing POST render template with current form_values
    return render_template_string(
        html_page,
        form_values=form_values,
        role=role,
        salary=salary,
        top3=top3,
        message=message,
        message_class=message_class,
        suggestions=suggestions_text,
        warning_text=warning_text,
        used_skills=used_skills
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
