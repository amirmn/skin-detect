import cv2
import numpy as np

def detect_skin_and_remove_background(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Image not found.")
        return None

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)  
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8) 

    skin_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    lower_background = np.array([247, 157, 105], dtype=np.uint8) 
    upper_background = np.array([247, 157, 105], dtype=np.uint8) 
    
    lower_background = np.clip(lower_background - 10, 0, 255)
    upper_background = np.clip(upper_background + 10, 0, 255)

    background_mask = cv2.inRange(img, lower_background, upper_background)

    img[background_mask == 255] = [0, 0, 0]  # سیاه کردن بک‌گراند

    output = np.zeros_like(img) 
    output[skin_mask == 255] = (255, 255, 255) 
   
    return output

output = detect_skin_and_remove_background("image_four.jpg")

if output is not None:
    cv2.imshow("Skin Detection and Background Removal", output)
    cv2.imwrite("skin_and_removed_background_output.jpg", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
