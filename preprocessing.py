# 1. Import libraries
import re  # للتعامل مع النصوص
import pandas as pd  # للجداول
import numpy as np
from sklearn.preprocessing import LabelEncoder  # يحول التصنيفات لأرقام
from sklearn.feature_extraction.text import TfidfVectorizer  # يحول النص لأرقام
from sklearn.model_selection import train_test_split  # تقسيم الداتا

# 2. Load the dataset and explore
# قراءة عمودين بس من الملف: نوع المنتج ونص الشكوى
df = pd.read_csv("data/rows.csv",
                 usecols=["Product", "Consumer complaint narrative"],
                 low_memory=False)
df.columns = ["label", "text"]  # تغيير الأسماء لأسماء أبسط
print(df.head())  # أول 5 صفوف
print(df.info())  # معلومات عن الأعمدة وعدد القيم الفاضية

# 3. Drop unnecessary rows
df = df.dropna()  # حذف الشكاوى اللي مفيهاش نص
print("Rows with text:", len(df))

# توحيد أسماء التصنيفات القديمة مع الجديدة
df["label"] = df["label"].replace({
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Bank account or service": "Checking or savings account",
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Consumer Loan": "Vehicle loan or lease",
})
df = df[df["label"] != "Other financial service"]  # حذف تصنيف غير واضح
print(df["label"].value_counts())  # عدد الشكاوى في كل تصنيف
print("Number of categories:", df["label"].nunique())  # عدد التصنيفات

# 4. Handle text (cleaning)
def clean_text(t):
    t = t.lower()  # حروف صغيرة
    t = re.sub(r"\bx{2,}\b", " ", t)  # حذف XXXX (بيانات مخفية)
    t = re.sub(r"[^a-z\s]", " ", t)  # حذف الأرقام والرموز
    t = re.sub(r"\s+", " ", t).strip()  # حذف المسافات الزيادة
    return t

df["text"] = df["text"].apply(clean_text)  # تطبيق الدالة على كل الشكاوى
df = df[df["text"].str.split().str.len() >= 5]  # نخلي الشكاوى اللي 5 كلمات أو أكتر
print("Rows after cleaning:", len(df))

# 5. Encode categorical data
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])  # كل تصنيف بياخد رقم من 0 لـ 8

# 6. Save the clean data
df.to_csv("data/clean.csv", index=False)  # حفظ الداتا النضيفة
print(df.head())

# 7. Define features (x) and target (y)
x = df["text"]  # المدخل: نص الشكوى
y = df["label_id"]  # المطلوب توقعه: رقم التصنيف

# 8. Split data into training and testing sets
x_train_text, x_test_text, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Feature extraction (TF-IDF) - fit on train only
vectorizer = TfidfVectorizer(max_features=5000)  # أهم 5000 كلمة
x_train = vectorizer.fit_transform(x_train_text)  # يتعلم من التدريب ويحول
x_test = vectorizer.transform(x_test_text)  # يحول الاختبار بس من غير ما يتعلم منه

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)
print("Data processing and splitting completed successfully!")
