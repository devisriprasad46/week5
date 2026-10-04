# MINI-LAB 1.5
# Zero-shot vs Few-shot Sentiment Classification

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-3.6-flash"


reviews = [
    ("The battery life is amazing!", "positive"),
    ("It works, nothing special.", "neutral"),
    ("The camera quality is terrible.", "negative"),
    ("I really love the display and performance.", "positive"),
    ("The phone is okay, but nothing impressive.", "neutral")
]


zero_results = []
few_results = []


for i, (review, actual) in enumerate(reviews, 1):

    prompt = f"""
Classify the following review using both zero-shot and few-shot prompting.

ZERO-SHOT:
Classify sentiment as positive, negative, or neutral.
Review: "{review}"
Answer:

FEW-SHOT:
Use these examples:
"Best phone ever!" → positive
"The camera is terrible." → negative
"The phone works as expected." → neutral

Review: "{review}"
Answer:

Return the answer in exactly this format:
Zero-shot: [sentiment]
Few-shot: [sentiment]
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    result = response.text.strip()

    print("\nReview", i, ":", review)
    print("Actual:", actual)
    print(result)

    # Save results
    lines = result.lower().splitlines()

    zero = ""
    few = ""

    for line in lines:
        if "zero-shot:" in line:
            zero = line.split(":")[-1].strip()

        if "few-shot:" in line:
            few = line.split(":")[-1].strip()

    zero_results.append(zero)
    few_results.append(few)


# Accuracy
zero_correct = 0
few_correct = 0

for i, (review, actual) in enumerate(reviews):

    if zero_results[i] == actual:
        zero_correct += 1

    if few_results[i] == actual:
        few_correct += 1


print("\n===================================")
print("COMPARISON SUMMARY")
print("===================================")

print("Method       Accuracy")
print("---------------------")
print("Zero-shot    ", zero_correct, "/5")
print("Few-shot     ", few_correct, "/5")


if few_correct > zero_correct:
    print("\nConclusion: Few-shot prompting performed better.")

elif few_correct == zero_correct:
    print("\nConclusion: Both methods achieved the same accuracy.")

else:
    print("\nConclusion: Zero-shot prompting performed better.")