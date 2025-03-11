import cv2
import numpy as np

def detect_skin(image_path):
    # بارگذاری تصویر
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Image not found.")
        return None

    # 1. تبدیل تصویر به سیاه و سفید (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. افزایش شدت رنگ (Contrast) با استفاده از اعمال تبدیل خطی
    # افزایش کنتراست
    enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)

    # 3. تبدیل تصویر به فضای رنگی HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. تعریف محدوده‌های رنگی پوست در فضای HSV
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)  # پایین‌ترین حد
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)  # بالاترین حد

    # 5. ساخت ماسک برای شناسایی پوست
    skin_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # 6. ایجاد یک تصویر سیاه با ابعاد همان تصویر ورودی
    output = np.zeros_like(img)

    # 7. اعمال ماسک پوست: قسمت‌های پوست سفید و بقیه سیاه
    output[skin_mask == 255] = (255, 255, 255)  # سفید کردن قسمت‌های پوست

    # 8. برگشت دادن تصویر خروجی
    return output

# # تست عملکرد تابع با یک تصویر نمونه
# output = detect_skin("image_four.jpg")

# # بررسی اینکه آیا تصویر با موفقیت پردازش شده است
# if output is not None:
#     cv2.imshow("Skin Detection Output", output)
#     cv2.imwrite("skin_detected_output.jpg", output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
