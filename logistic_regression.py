# 1. Imports and loading
import numpy as np  # للعمليات الحسابية والأرقام
import pandas as pd  # للجداول
import matplotlib.pyplot as plt  # للرسومات
from sklearn.model_selection import train_test_split  # تقسيم الداتا
from sklearn.feature_extraction.text import TfidfVectorizer  # يحول النص لأرقام
from sklearn.linear_model import LogisticRegression  # موديل التصنيف

# قراءة الداتا النضيفة اللي طلعت من خطوة الـ preprocessing
df = pd.read_csv("data/clean.csv")
X = df["text"]  # المدخل: نص الشكوى
y = df["label"]  # المطلوب توقعه: القسم

print("Shape:", df.shape)  # عدد الصفوف والأعمدة
print("Classes:", y.unique())  # الأقسام الموجودة
print("\nClass counts:")
print(df["label"].value_counts())  # عدد الشكاوى في كل قسم
print("\nMissing values:", df.isnull().sum().sum())  # عدد الخانات الفاضية (المفروض صفر)
df.head()  # أول 5 صفوف


# 2. Explore and Visualize
# رسمة أعمدة بتوضح عدد الشكاوى في كل قسم (عشان نشوف توازن الداتا)
counts = df["label"].value_counts().sort_values()  # ترتيب الأقسام من الأصغر للأكبر
plt.figure(figsize=(10, 5))
plt.barh(counts.index, counts.values, color="blue", alpha=0.6)
plt.xlabel("Number of samples")
plt.title("Class Distribution")
plt.tight_layout()  # ضبط المسافات عشان الأسماء الطويلة متتقصش
plt.show()

# رسمة بتوضح طول الشكاوى بعدد الكلمات
# clip(upper=500) بيحسب أي شكوى أطول من 500 كلمة كأنها 500 عشان الرسمة متتمطش
plt.figure(figsize=(7, 5))
plt.hist(df["text"].str.split().str.len().clip(upper=500), bins=50, color="red", alpha=0.6)
plt.xlabel("Number of words")
plt.ylabel("Number of complaints")
plt.title("Complaint Length")
plt.grid(True)
plt.show()


# 3. Train/Test Split and Feature Extraction
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", len(X_train_text))  # حجم التدريب
print("Test size: ", len(X_test_text))  # حجم الاختبار

# TF-IDF: يتعلم من التدريب بس، وبعدين يحول التدريب والاختبار (عشان منعملش Data Leakage)
vectorizer = TfidfVectorizer(max_features=5000)  # أهم 5000 كلمة
X_train_scaled = vectorizer.fit_transform(X_train_text)  # يتعلم من التدريب ويحول
X_test_scaled = vectorizer.transform(X_test_text)  # يحول الاختبار من غير ما يتعلم منه

print("\nTrain shape:", X_train_scaled.shape)  # شكل مصفوفة التدريب
print("Test shape: ", X_test_scaled.shape)  # شكل مصفوفة الاختبار


# 4. Train the Model and Predict
model = LogisticRegression(max_iter=300)  # إنشاء الموديل (أقصى 300 محاولة لتحسين الأوزان)
model.fit(X_train_scaled, y_train)  # تدريب الموديل على بيانات التدريب
print("Training done!")
print("Number of classes:", len(model.classes_))  # عدد الأقسام (9)
print("Weights shape:", model.coef_.shape)  # شكل مصفوفة الأوزان: (أقسام × كلمات)

y_pred = model.predict(X_test_scaled)  # القسم المتوقع لكل شكوى
y_prob = model.predict_proba(X_test_scaled)  # احتمال كل قسم من الأقسام الـ 9

# جدول بيقارن الحقيقي بالمتوقع وثقة الموديل لأول 10 شكاوى
results = pd.DataFrame({
    "Actual": y_test.values[:10],  # القسم الحقيقي
    "Predicted": y_pred[:10],  # القسم المتوقع
    "Max Prob": y_prob[:10].max(axis=1).round(3)  # أعلى احتمال (ثقة الموديل)
})
print(results)
print("\nAccuracy:", round(model.score(X_test_scaled, y_test), 3))  # نسبة التوقعات الصح


# 5. Evaluation: Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # مصفوفة الخلط

# الصفوف = القسم الحقيقي، والأعمدة = القسم المتوقع، والقطر = التوقعات الصح
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)
disp = ConfusionMatrixDisplay(cm, display_labels=range(len(model.classes_)))  # أرقام بدل الأسماء عشان الرسمة تتقري
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
# طباعة مفتاح الأرقام (كل رقم لأنهي قسم)
for i, c in enumerate(model.classes_):
    print(i, "=", c)


# 6. Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report)  # مقاييس التقييم

print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))  # نسبة التوقعات الصح
# average="weighted" بيحسب المتوسط مع إعطاء القسم الكبير وزن أكبر
print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 3))  # لما الموديل يقول قسم، كام % صح
print("Recall:   ", round(recall_score(y_test, y_pred, average="weighted"), 3))  # من الشكاوى الحقيقية، كام % لقاها
print("F1:       ", round(f1_score(y_test, y_pred, average="weighted"), 3))  # المتوسط بين Precision و Recall
print("\nFull report:")
print(classification_report(y_test, y_pred))  # تقرير مفصل لكل قسم


# 7. ROC Curve and AUC (One-vs-Rest)
from sklearn.metrics import roc_curve, roc_auc_score  # أدوات منحنى ROC
from sklearn.preprocessing import label_binarize  # يحول الأقسام لأعمدة 0/1

# One-vs-Rest: لكل قسم نسأل "الشكوى دي من القسم ده ولا لأ؟"
y_test_bin = label_binarize(y_test, classes=model.classes_)
auc = roc_auc_score(y_test_bin, y_prob, average="macro")  # متوسط AUC لكل الأقسام بالتساوي
print(f"AUC (macro) = {auc:.4f}")

# رسم منحنى ROC لكل قسم
plt.figure(figsize=(8, 6))
for i in range(len(model.classes_)):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_test_bin[:, i], y_prob[:, i])  # حساب نقط المنحنى للقسم ده
    plt.plot(fpr_arr, tpr_arr, linewidth=1.5, label=f"Class {i}")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")  # الخط المنقط: التخمين العشوائي
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC Curve (macro AUC = {auc:.3f})")
plt.legend()
plt.grid(True)
plt.show()