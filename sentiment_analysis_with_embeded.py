# install neccery libs
!pip install pandas numpy scikit-learn nltk gensim
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import KeyedVectors
nltk.download('stopwords')
from nltk.corpus import stopwords

# نمایش چند کلمه stopword
print(stopwords.words('english')[:5])
# خواندن دیتاست با ستون‌های مناسب
file_path = 'training.1600000.processed.noemoticon.csv'
column_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']

df = pd.read_csv(
    file_path,
    header=0,  # سطر اول header هست
    encoding='latin1',
    names=column_names,
    low_memory=False
)

# فیلتر فقط برای 0 و 4
df = df[df['sentiment'].isin([0, 4])]

print("✅ تعداد داده‌ها:", len(df))
print("توزیع کلاس‌ها:")
print(df['sentiment'].value_counts())
from nltk.corpus import stopwords
import re

# تنظیم stop words + کلمات اضافی
stop_words = set(stopwords.words('english'))
extra_stopwords = {'rt', 'user', 'url', '…', 'amp', 'via', 'http', 'https'}
all_stopwords = stop_words.union(extra_stopwords)

# تابع پیش‌پردازش بهتر شده
def preprocess_text_improved(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in all_stopwords]
    return filtered_tokens

# اعمال پیش‌پردازش
df['tokens'] = df['text'].apply(preprocess_text_improved)
import gensim.downloader as api

# دانلود مدل "glove-twitter-25" - حدود 86 مگابایت
embedding_model = api.load("glove-twitter-25")

print("✅ مدل embedding با موفقیت دانلود و بارگذاری شد!")


import numpy as np

# تابع برای تبدیل tokens به بردار (با استفاده از GloVe)
def text_to_vector(tokens, model, vector_size=25):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# اعمال تابع برای تمام داده‌ها
X = np.array(df['tokens'].apply(text_to_vector, model=embedding_model).tolist())
y = df['sentiment'].map({0: 0, 4: 1})  # 0 = منفی، 1 = مثبت

print("✅ ابعاد X:", X.shape)
print("✅ ابعاد y:", y.shape)



from sklearn.model_selection import train_test_split

# تقسیم داده به train و test
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,     # 80% train, 20% test
    random_state=42,   # ثابت کردن رندوم برای قابلیت تکرار
    stratify=y         # حفظ توزیع کلاس‌ها در train و test
)

# چاپ ابعاد
print("✅ X_train shape:", X_train.shape)
print("✅ X_test shape:", X_test.shape)
print("✅ y_train shape:", y_train.shape)
print("✅ y_test shape:", y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ساخت و آموزش مدل
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# پیش‌بینی روی test
y_pred = model.predict(X_test)

# محاسبه دقت و گزارش طبقه‌بندی
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 دقت مدل: {accuracy:.4f}")
print("\n📋 گزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# محاسبه ماتریس اشتباه
cm = confusion_matrix(y_test, y_pred)

# رسم ماتریس اشتباه
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

from sklearn.model_selection import cross_val_score
import numpy as np

# اجرای 5-Fold Cross Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# چاپ نتایج
print("🎯 دقت در هر Fold:")
print(cv_scores)
print(f"📌 میانگین دقت: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")