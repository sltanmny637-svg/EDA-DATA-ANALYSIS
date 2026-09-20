import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ضبط شكل الرسوم البيانية
sns.set_theme(style="whitegrid")

# ==========================================
# 1. تحميل البيانات (Load Dataset)
# ==========================================
# استخدام رابط مباشر لرابط بيانات السيارات الشهير (Automobile Dataset)
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Datafiles/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, names=headers)

# استبدال العلامة "?" بـ NaN لتسهيل المعالجة
df.replace("?", np.nan, inplace=True)

# ==========================================
# 2. Data Inspection & Cleaning
# ==========================================
print("--- 1. Data Inspection ---")
df.info()
print("\n--- Descriptive Statistics ---")
print(df.describe())

# تحويل الأعمدة إلى أنواع رقمية (Numeric Types)
numeric_cols = ['horsepower', 'price', 'curb-weight', 'engine-size']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

# التعامل مع القيم المفقودة (Null Values) في عمود horsepower باستخدام الوسيط (Median)
median_hp = df['horsepower'].median()
df['horsepower'].fillna(median_hp, inplace=True)

# حذف الصفوف التي لا تحتوي على سعر (price)
df.dropna(subset=['price'], inplace=True)

# تقسيم البيانات إلى فئات (Binning) لتصنيف horsepower إلى (Low, Medium, High)
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)

print("\n--- Horsepower Categories Count ---")
print(df['horsepower-binned'].value_counts())

# ==========================================
# 3. Univariate Analysis
# ==========================================
# أ) تحليل التوزيع التكراري للأسعار (Distribution of Price)
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True, color='skyblue', bins=20)
plt.title('Distribution of Price', fontsize=14, fontweight='bold')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# ب) دراسة توزيع أنواع هياكل السيارات (body-style)
plt.figure(figsize=(8, 5))
sns.countplot(x='body-style', data=df, palette='viridis', order=df['body-style'].value_counts().index)
plt.title('Distribution of Body Style', fontsize=14, fontweight='bold')
plt.xlabel('Body Style')
plt.ylabel('Count')
plt.show()

# ==========================================
# 4. Bivariate & Multivariate Analysis
# ==========================================
# أ) إنشاء Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.show()

# ب) استخدام Scatter Plots لاستكشاف العلاقات المباشرة مع السعر
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(x='engine-size', y='price', data=df, ax=axes[0], color='blue', alpha=0.7)
axes[0].set_title('Engine Size vs Price', fontweight='bold')

sns.scatterplot(x='horsepower', y='price', data=df, ax=axes[1], color='green', alpha=0.7)
axes[1].set_title('Horsepower vs Price', fontweight='bold')

sns.scatterplot(x='curb-weight', y='price', data=df, ax=axes[2], color='red', alpha=0.7)
axes[2].set_title('Curb Weight vs Price', fontweight='bold')

plt.tight_layout()
plt.show()

# ج) Pairplot لاستكشاف جميع العلاقات التبادلية
sns.pairplot(df[numeric_cols], diag_kind='kde')
plt.suptitle('Pairplot of Key Variables', y=1.02, fontsize=14, fontweight='bold')
plt.show()
