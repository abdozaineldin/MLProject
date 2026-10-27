# 1. Import libraries
import re  # للتعامل مع النصوص
import pandas as pd  # للجداول
import numpy as np  # للعمليات الحسابية والأرقام
import matplotlib.pyplot as plt  # للرسومات
from scipy.sparse import hstack  # يلزق مصفوفتين جنب بعض
from sklearn.feature_extraction.text import TfidfVectorizer  # يحول النص لأرقام
from sklearn.model_selection import train_test_split  # تقسيم الداتا
from sklearn.linear_model import LinearRegression  # موديل التنبؤ بالأرقام
from sklearn.metrics import mean_squared_error, r2_score  # مقاييس التقييم

# 2. Load the dataset
# قراءة 4 أعمدة بس من الملف: تاريخ الاستلام وتاريخ الإرسال وطريقة التقديم ونص الشكوى
df = pd.read_csv("data/rows.csv",
                 usecols=["Date received", "Date sent to company",
                          "Submitted via", "Consumer complaint narrative"],
                 low_memory=False)
df = df.dropna()  # حذف الصفوف اللي فيها خانات فاضية

# 3. Handle text (cleaning)
def clean_text(t):
    t = t.lower()  # حروف صغيرة
    t = re.sub(r"\bx{2,}\b", " ", t)  # حذف XXXX (بيانات مخفية)
    t = re.sub(r"[^a-z\s]", " ", t)  # حذف الأرقام والرموز
    return re.sub(r"\s+", " ", t).strip()  # حذف المسافات الزيادة

df["text"] = df["Consumer complaint narrative"].apply(clean_text)  # تطبيق الدالة على كل الشكاوى

# 4. Create the target (days)
# حساب عدد الأيام بين استلام الشكوى وإرسالها للشركة (ده اللي عايزين نتنبأ بيه)
df["days"] = (pd.to_datetime(df["Date sent to company"])
              - pd.to_datetime(df["Date received"])).dt.days
print(df.head())  # أول 5 صفوف

# 5. Check dataset info
print(df.info())  # معلومات عن الأعمدة وعدد القيم الفاضية
print(df["days"].describe())  # إحصائيات عن عدد الأيام (المتوسط والأكبر والأصغر)

# 6. Remove wrong/extreme values
# نشيل القيم السالبة والقيم الشاذة (أكتر من 30 يوم) لأنها بتبوّظ الموديل
df = df[(df["days"] >= 0) & (df["days"] <= 30)]

# 7. Define features (X) and target (y)
vectorizer = TfidfVectorizer(max_features=3000)  # أهم 3000 كلمة
X_text = vectorizer.fit_transform(df["text"])  # تحويل نص الشكوى لأرقام
X_via = pd.get_dummies(df["Submitted via"]).values.astype(float)  # تحويل طريقة التقديم لأعمدة 0/1
X = hstack([X_text, X_via]).tocsr()  # لزق النص مع طريقة التقديم في مصفوفة واحدة
y = df["days"]  # المطلوب توقعه: عدد الأيام

# 8. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Create and train the Linear Regression model
model = LinearRegression()  # إنشاء الموديل
model.fit(X_train, y_train)  # تدريب الموديل على بيانات التدريب

# 10. Make predictions
y_pred = model.predict(X_test)  # توقع عدد الأيام لبيانات الاختبار

# 11. Evaluate model performance
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")  # متوسط مربع الخطأ (كل ما قل كان أحسن)
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")  # نسبة التغير اللي الموديل فسّرها (1 هو الأفضل)

# 12. Intercept and coefficients
print("Intercept:", model.intercept_)  # الرقم الثابت اللي الموديل بيبدأ منه
# أسماء الميزات: كلمات النص + طرق التقديم
names = list(vectorizer.get_feature_names_out()) + list(pd.get_dummies(df["Submitted via"]).columns)
coefficients = pd.DataFrame(model.coef_, names, columns=["Coefficient"])  # وزن كل ميزة
print(coefficients.sort_values("Coefficient", ascending=False).head(10))  # أهم 10 ميزات

# 13. Visualization (Distribution of Days)
# رسمة توضح توزيع عدد الأيام (أغلبها صفر)
plt.hist(df["days"], bins=30, alpha=0.6)
plt.xlabel("Days until sent to company")
plt.ylabel("Number of complaints")
plt.title("Distribution of Days")
plt.show()

# 14. Predicted vs Actual (sample of 3000 points)
# رسمة بتقارن القيمة الحقيقية بتوقع الموديل على عينة 3000 شكوى
idx = np.random.RandomState(42).choice(len(y_test), 3000, replace=False)
plt.scatter(y_test.values[idx], y_pred[idx], alpha=0.3, color="purple")
plt.xlabel("Actual Days")
plt.ylabel("Predicted Days")
plt.title("Actual vs Predicted Days")
plt.show()