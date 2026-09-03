# generate_traffic.py
# Sends test requests to populate the database with real inspection data.
# Run this after starting the API to generate meaningful statistics.

import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

# Get test images
test_images = (
    list(Path("data/raw/test/images").glob("*.jpg")) +
    list(Path("data/raw/test/images").glob("*.png"))
)[:50]  # use first 50 images

print(f"Sending {len(test_images)} inspection requests...")
print("="*60)

passed = 0
failed = 0
uncertain = 0
total_defects = 0

for i, img_path in enumerate(test_images):
    with open(img_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/inspect",
            files={"file": (img_path.name, f, "image/jpeg")},
            params={"return_image": False}
        )

    if response.status_code == 200:
        data = response.json()
        status = data["quality_status"]
        defects = data["total_defects"]
        total_defects += defects

        if status == "PASS":
            passed += 1
            icon = "✅"
        elif status == "UNCERTAIN":
            uncertain += 1
            icon = "⚠"
        else:
            failed += 1
            icon = "❌"

        print(
            f"[{i+1:3d}] {icon} {status:10} "
            f"Defects: {defects} | "
            f"{data['defect_summary'][:50]}"
        )
    else:
        print(f"[{i+1:3d}] Error: {response.status_code}")

    time.sleep(0.3)

print("\n" + "="*60)
print("TRAFFIC GENERATION COMPLETE")
print("="*60)
print(f"Total inspected:  {len(test_images)}")
print(f"Passed:           {passed}")
print(f"Failed:           {failed}")
print(f"Uncertain:        {uncertain}")
print(f"Total defects:    {total_defects}")
print(f"Yield rate:       {passed/len(test_images):.1%}")

print("\nNow testing quality agent queries...")
print("="*60)

questions = [
    "What is our overall yield rate?",
    "Which defect type is most common?",
    "What quality improvements do you recommend?",
    "How many PCBs failed inspection today?",
]

for question in questions:
    response = requests.post(
        f"{API_URL}/query",
        json={"question": question, "history": []}
    )

    if response.status_code == 200:
        answer = response.json()["answer"]
        print(f"\nQ: {question}")
        print(f"A: {answer[:200]}...")
    else:
        print(f"Query error: {response.status_code}")

    time.sleep(1)

print("\nDone. Check http://localhost:8000/docs")