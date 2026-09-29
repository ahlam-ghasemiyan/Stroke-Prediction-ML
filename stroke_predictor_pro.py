

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from PIL import Image, ImageTk, ImageFilter
import warnings
warnings.filterwarnings('ignore')

# ۱. آموزش و بارگذاری مدل‌ها
def train_and_save_models():
    df = pd.read_csv('stroke_dataset.csv')
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    numeric_cols = ['age', 'bmi', 'avg_glucose_level']
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
    }
    trained_models = {}
    results = {}
    for name, model in models.items():
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        y_pred = pipeline.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred)
        }
    joblib.dump(trained_models, 'stroke_models.pkl')
    joblib.dump(results, 'model_results.pkl')
    return trained_models, results

try:
    models = joblib.load('stroke_models.pkl')
    results = joblib.load('model_results.pkl')
    print("✅ مدل‌ها بارگذاری شدند.")
except:
    print("⚠️ مدل‌ها یافت نشدند. در حال آموزش مجدد...")
    models, results = train_and_save_models()

# ۲. کلاس اصلی برنامه
class StrokePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 پیش‌بینی خطر سکته مغزی")
        self.root.geometry("950x850")
        self.root.resizable(False, False)

        # تنظیم پس‌زمینه با بلور
        self.set_background()

        self.history = []
        self.load_history()
        self.create_widgets()

    def set_background(self):
        """بارگذاری تصویر با بلور کم به‌عنوان پس‌زمینه"""
        try:
            image = Image.open("backg (1).png")
            image = image.resize((950, 750), Image.Resampling.LANCZOS)
            blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
            self.bg_image = ImageTk.PhotoImage(blurred)
            self.bg_label = tk.Label(self.root, image=self.bg_image)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            print("✅ پس‌زمینه با بلور بارگذاری شد.")
        except:
            self.root.configure(bg='#f0f4f8')
            print("⚠️ تصویر یافت نشد، پس‌زمینه ساده استفاده شد.")

    def create_widgets(self):
        # عنوان 
        title_label = tk.Label(self.root, text="🧠 پیش‌بینی خطر سکته مغزی",
                               font=("B Titr", 18, "bold"), bg='white', fg='#2c3e50')
        title_label.pack(pady=15)

        # کادر ورودی (دو ستون) 
        main_frame = tk.Frame(self.root, bg='white', bd=2, relief=tk.RIDGE)
        main_frame.pack(padx=20, pady=10, fill=tk.X)

        # تعریف فیلدها (نام برچسب، کلید، مقدار پیش‌فرض، نوع)
        fields = [
            ("سن (سال):", "entry_age", "50", "entry"),
            ("جنسیت:", "combo_gender", ["Male", "Female"], "combo"),
            ("فشار خون:", "combo_hypertension", ["بله", "خیر"], "combo"),
            ("بیماری قلبی:", "combo_heart_disease", ["بله", "خیر"], "combo"),
            ("وضعیت تأهل:", "combo_married", ["Yes", "No"], "combo"),
            ("نوع شغل:", "combo_work", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], "combo"),
            ("نوع محل سکونت:", "combo_residence", ["Urban", "Rural"], "combo"),
            ("قند خون متوسط:", "entry_glucose", "95", "entry"),
            ("شاخص توده بدنی (BMI):", "entry_bmi", "25", "entry"),
            ("وضعیت سیگار:", "combo_smoking", ["never", "formerly", "smokes", "Unknown"], "combo")
        ]

        self.entries = {}
        row = 0
        col = 0

        for label_text, key, value, ftype in fields:
            frame = tk.Frame(main_frame, bg='white')
            frame.grid(row=row, column=col, sticky='ew', padx=10, pady=5)

            lbl = tk.Label(frame, text=label_text, font=("B Nazanin", 11), bg='white', width=16, anchor='w')
            lbl.pack(side=tk.LEFT)

            if ftype == "entry":
                widget = tk.Entry(frame, font=("Arial", 11), width=14)
                widget.insert(0, value)
            else:
                widget = ttk.Combobox(frame, values=value, width=12, state="readonly")
                widget.set(value[0] if isinstance(value, list) else value)

            widget.pack(side=tk.RIGHT)
            self.entries[key] = widget

            # حرکت به ستون بعدی یا سطر بعدی
            if col == 0:
                col = 1
            else:
                col = 0
                row += 1

        # تنظیم کشش ستون‌ها
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        #  دکمه‌ها
        btn_frame = tk.Frame(self.root, bg='white')
        btn_frame.pack(pady=10)

        # دکمه پیش‌بینی اصلی
        btn_predict = tk.Button(btn_frame, text="🔍 پیش‌بینی خطر سکته",
                                font=("B Nazanin", 13, "bold"),
                                bg='#3498db', fg='white', padx=20, pady=8,
                                command=self.predict)
        btn_predict.grid(row=0, column=0, columnspan=3, pady=5)

        # دکمه‌های تحلیلی
        btns = [
            ("📊 ماتریس درهم‌ریختگی", self.show_confusion_matrices, '#9b59b6'),
            ("📈 بررسی کلی دیتاست", self.show_dataset_overview, '#2ecc71'),
            ("📋 مقایسه مدل‌ها", self.show_model_comparison, '#e67e22'),
            ("📜 تاریخچه پیش‌بینی‌ها", self.show_history, '#1abc9c'),
            ("📄 صادرات گزارش", self.export_report, '#34495e'),
            ("🔍 بهبود مدل", self.optimize_models, '#e74c3c')
        ]

        for i, (text, cmd, color) in enumerate(btns):
            btn = tk.Button(btn_frame, text=text, font=("B Nazanin", 10),
                            bg=color, fg='white', padx=10, pady=5, command=cmd)
            btn.grid(row=1 + i//3, column=i % 3, padx=5, pady=3, sticky='ew')

        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)

       
        #  بخش نمایش نتیجه (دو ستونی)
        result_frame = tk.Frame(self.root, bg='white', bd=2, relief=tk.RIDGE)
        result_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        #  ستون‌بندی نتیجه 
        result_main = tk.Frame(result_frame, bg='white')
        result_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ستون چپ (سطح خطر + توصیه‌ها)
        left_col = tk.Frame(result_main, bg='white')
        left_col.grid(row=0, column=0, sticky='nsew', padx=5)

        # ستون راست (احتمال سکته + نتیجه اصلی)
        right_col = tk.Frame(result_main, bg='white')
        right_col.grid(row=0, column=1, sticky='nsew', padx=5)

        # تنظیم وزن ستون‌ها (برای کشش یکسان)
        result_main.grid_columnconfigure(0, weight=1)
        result_main.grid_columnconfigure(1, weight=1)
        result_main.grid_rowconfigure(0, weight=1)

        #  ستون چپ 
        # سطح خطر (بالا)
        self.label_risk = tk.Label(left_col, text="سطح خطر: -",
                                   font=("B Nazanin", 14, "bold"), fg='#d35400', bg='white')
        self.label_risk.pack(pady=5)

        #  توصیه‌ها 
        self.text_details = tk.Text(left_col, height=8, width=40,
                                    font=("B Nazanin", 11), bg='#f8f9fa',
                                    wrap=tk.WORD, state=tk.DISABLED,
                                    relief=tk.FLAT, padx=10, pady=10)
        self.text_details.pack(fill=tk.BOTH, expand=True, pady=5)

        #  ستون راست 
        # احتمال سکته (بالا)
        self.label_prob = tk.Label(right_col, text="احتمال سکته: ۰٪",
                                   font=("B Nazanin", 14, "bold"), fg='#2980b9', bg='white')
        self.label_prob.pack(pady=5)

        # نتیجه اصلی 
        self.label_result = tk.Label(right_col, text="نتیجه پیش‌بینی",
                                     font=("B Titr", 18, "bold"), fg='#2c3e50', bg='white')
        self.label_result.pack(expand=True, pady=10)

    def get_input_data(self):
        data = {
            'age': float(self.entries['entry_age'].get()),
            'gender': self.entries['combo_gender'].get(),
            'hypertension': 1 if self.entries['combo_hypertension'].get() == "بله" else 0,
            'heart_disease': 1 if self.entries['combo_heart_disease'].get() == "بله" else 0,
            'ever_married': self.entries['combo_married'].get(),
            'work_type': self.entries['combo_work'].get(),
            'Residence_type': self.entries['combo_residence'].get(),
            'avg_glucose_level': float(self.entries['entry_glucose'].get()),
            'bmi': float(self.entries['entry_bmi'].get()),
            'smoking_status': self.entries['combo_smoking'].get()
        }
        return data

    def predict(self):
        try:
            data = self.get_input_data()
            input_df = pd.DataFrame([data])
            model = models['Logistic Regression']
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            if prediction == 1:
                result_text = "⚠️ خطر سکته وجود دارد!"
                result_color = "red"
                risk_level = "بالا" if probability > 0.7 else "متوسط"
            else:
                result_text = "✅ خطر سکته وجود ندارد."
                result_color = "green"
                risk_level = "کم"
            self.label_result.config(text=result_text, fg=result_color)
            self.label_prob.config(text=f"احتمال سکته: {probability:.2%}")
            self.label_risk.config(text=f"سطح خطر: {risk_level}")
            recommendations = []
            if data['age'] > 60: recommendations.append("• سن بالا: معاینات منظم قلبی و عروقی")
            if data['bmi'] > 30: recommendations.append("• BMI بالا: کاهش وزن و تغذیه مناسب")
            if data['avg_glucose_level'] > 140: recommendations.append("• قند خون بالا: کنترل دیابت")
            if data['hypertension'] == 1: recommendations.append("• فشار خون بالا: مصرف منظم دارو")
            if data['smoking_status'] in ['smokes', 'formerly']: recommendations.append("• ترک سیگار")
            if not recommendations: recommendations.append("• وضعیت شما نسبتاً خوب است.")
            detail = f"📊 نتیجه پیش‌بینی:\n----------------------------------------\nوضعیت: {result_text}\nاحتمال سکته: {probability:.2%}\nسطح خطر: {risk_level}\n----------------------------------------\n📌 توصیه‌ها:\n" + "\n".join(recommendations)
            self.text_details.config(state=tk.NORMAL)
            self.text_details.delete(1.0, tk.END)
            self.text_details.insert(tk.END, detail)
            self.text_details.config(state=tk.DISABLED)
            self.history.append({
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **data,
                'prediction': int(prediction),
                'probability': round(probability, 4)
            })
            self.save_history()
        except Exception as e:
            messagebox.showerror("خطا", f"ورودی نامعتبر!\n{str(e)}")

    # متد  
    def show_confusion_matrices(self):
        try:
            df = pd.read_csv('stroke_dataset.csv')
            if 'id' in df.columns:
                df.drop('id', axis=1, inplace=True)
            X = df.drop('stroke', axis=1)
            y = df['stroke']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('ماتریس درهم‌ریختگی - مقایسه مدل‌ها', fontsize=14)
            for idx, (name, model) in enumerate(models.items()):
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                            xticklabels=['No Stroke', 'Stroke'],
                            yticklabels=['No Stroke', 'Stroke'])
                axes[idx].set_title(name)
                axes[idx].set_ylabel('واقعی')
                axes[idx].set_xlabel('پیش‌بینی')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در نمایش ماتریس: {e}")

    def show_dataset_overview(self):
        try:
            df = pd.read_csv('stroke_dataset.csv')
            if 'id' in df.columns:
                df.drop('id', axis=1, inplace=True)
            stats = f"""
📊 **بررسی کلی دیتاست** (Stroke Prediction Dataset)
========================================
🔹 تعداد نمونه‌ها: {df.shape[0]} رکورد
🔹 تعداد ویژگی‌ها: {df.shape[1]} ستون
🔹 ویژگی‌های عددی: {df.select_dtypes(include=['int64', 'float64']).shape[1]} عدد
🔹 ویژگی‌های دسته‌ای: {df.select_dtypes(include=['object']).shape[1]} دسته
🔹 داده‌های گمشده: {df.isnull().sum().sum()} عدد
🔹 ستون‌های با داده گمشده: {', '.join([col for col in df.columns if df[col].isnull().sum()>0]) or 'ندارد'}

📈 **توزیع کلاس هدف (stroke):**
   • بدون سکته: {df['stroke'].value_counts()[0]} ({df['stroke'].value_counts()[0]/len(df)*100:.2f}%)
   • سکته: {df['stroke'].value_counts().get(1, 0)} ({df['stroke'].value_counts().get(1,0)/len(df)*100:.2f}%)

📐 **آمار توصیفی:**
   • سن: میانگین={df['age'].mean():.1f}, انحراف={df['age'].std():.1f}
   • BMI: میانگین={df['bmi'].mean():.1f}, انحراف={df['bmi'].std():.1f}
   • قند خون: میانگین={df['avg_glucose_level'].mean():.1f}, انحراف={df['avg_glucose_level'].std():.1f}
"""
            messagebox.showinfo("بررسی کلی دیتاست", stats)
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در بررسی دیتاست: {e}")

    def show_model_comparison(self):
        try:
            df = pd.read_csv('stroke_dataset.csv')
            if 'id' in df.columns:
                df.drop('id', axis=1, inplace=True)
            X = df.drop('stroke', axis=1)
            y = df['stroke']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            comparison = "📋 **مقایسه عملکرد مدل‌ها**\n========================================\n"
            comparison += f"{'مدل':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
            comparison += "-" * 75 + "\n"
            for name, model in models.items():
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                comparison += f"{name:<25} {acc:.4f}     {prec:.4f}     {rec:.4f}     {f1:.4f}\n"
            messagebox.showinfo("مقایسه مدل‌ها", comparison)
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در مقایسه: {e}")

    def show_history(self):
        if not self.history:
            messagebox.showinfo("تاریخچه", "هیچ پیش‌بینی ذخیره‌ای یافت نشد.")
            return
        history_text = "📜 **تاریخچه پیش‌بینی‌ها**\n" + "="*50 + "\n"
        for i, entry in enumerate(self.history[-10:]):
            history_text += f"{i+1}. {entry['datetime']}\n"
            history_text += f"   سن: {entry['age']}, BMI: {entry['bmi']}, قند: {entry['avg_glucose_level']}\n"
            history_text += f"   نتیجه: {'سکته' if entry['prediction']==1 else 'بدون سکته'} (احتمال: {entry['probability']*100:.1f}%)\n"
            history_text += "-"*50 + "\n"
        messagebox.showinfo("تاریخچه پیش‌بینی‌ها", history_text)

    def export_report(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="ذخیره گزارش"
        )
        if filename:
            if self.history:
                df_report = pd.DataFrame(self.history)
                df_report.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("موفق", f"گزارش در {filename} ذخیره شد.")
            else:
                messagebox.showwarning("هشدار", "هیچ داده‌ای برای صادرات وجود ندارد.")

    def optimize_models(self):
        messagebox.showinfo("بهبود مدل", "این عملیات زمان‌بر است و بهینه‌سازی Hyperparameterها را انجام می‌دهد.")
        try:
            df = pd.read_csv('stroke_dataset.csv')
            if 'id' in df.columns:
                df.drop('id', axis=1, inplace=True)
            X = df.drop('stroke', axis=1)
            y = df['stroke']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            numeric_cols = ['age', 'bmi', 'avg_glucose_level']
            categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ])
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l2']
            }
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            messagebox.showinfo("بهبود مدل", f"بهترین امتیاز F1: {best_score:.4f}\nبهترین پارامترها: {best_params}")
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در بهبود مدل: {e}")

    def load_history(self):
        if os.path.exists('history.json'):
            try:
                with open('history.json', 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except:
                self.history = []

    def save_history(self):
        try:
            with open('history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except:
            pass

# ۳. اجرای برنامه
if __name__ == "__main__":
    root = tk.Tk()
    app = StrokePredictorApp(root)
    root.mainloop()


#ساخته شده توسط احلام قاسمیان برای اریه درس هوش مصنوعی
#  مقطع ارشد رشته هوش مصنوعی در تاریخ 30 تیر ماه 1405
#استاد مربوط = جناب دکتر مسعود دادگر