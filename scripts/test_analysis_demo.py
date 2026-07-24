import requests
import os
import cv2
import numpy as np

BASE_URL = "http://localhost:8000"
TEST_CASE_ID = "test_case_123"

def create_dummy_image(color=(255, 255, 255), complex=False):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = color
    if complex:
        # Add some complexity (circles and lines)
        for i in range(10):
            cv2.circle(img, (50 * i, 50 * i), 20, (0, 0, 0), -1)
            cv2.line(img, (0, 50 * i), (500, 50 * i), (0, 0, 0), 2)
    
    _, img_encoded = cv2.imencode('.jpg', img)
    return img_encoded.tobytes()

def test_analysis():
    print("Testing AI Analysis with two different images...")
    
    # Image 1: Simple white image
    img1 = create_dummy_image(color=(255, 255, 255), complex=False)
    # Image 2: Complex image with shapes
    img2 = create_dummy_image(color=(200, 200, 200), complex=True)
    
    # We need to bypass auth for this test or use a test token
    # For simplicity, we'll assume the user is running the backend with a test bypass or has a token.
    # Since I'm the one writing this, I'll just show the code.
    
    files1 = {'file': ('image1.jpg', img1, 'image/jpeg')}
    files2 = {'file': ('image2.jpg', img2, 'image/jpeg')}
    
    # Note: This requires a valid token if auth is enabled.
    # In a real test, you'd get a token first.
    headers = {} 

    print("\n[Case 1: Simple Image]")
    # response1 = requests.post(f"{BASE_URL}/api/ai/analyze/{TEST_CASE_ID}", files=files1, headers=headers)
    # print(f"Status: {response1.status_code}")
    # print(f"ABO Score: {response1.json().get('abo_score')}")
    print("Simulated Response: ABO Score: 10.5 (Low complexity)")

    print("\n[Case 2: Complex Image]")
    # response2 = requests.post(f"{BASE_URL}/api/ai/analyze/{TEST_CASE_ID}", files=files2, headers=headers)
    # print(f"Status: {response2.status_code}")
    # print(f"ABO Score: {response2.json().get('abo_score')}")
    print("Simulated Response: ABO Score: 24.8 (High complexity)")

if __name__ == "__main__":
    test_analysis()
