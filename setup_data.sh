#!/bin/bash
# Download the 240 extraction questions from the assistant-axis repo.
# Run this once before training.

set -e

QUESTIONS_URL="https://raw.githubusercontent.com/safety-research/assistant-axis/master/data/extraction_questions.jsonl"
OUTPUT="data/questions.jsonl"

mkdir -p data

echo "Downloading extraction questions..."
curl -fSL "$QUESTIONS_URL" -o "$OUTPUT"

# Check format and count
N=$(wc -l < "$OUTPUT")
echo "Downloaded $N questions to $OUTPUT"
echo "First line:"
head -1 "$OUTPUT"
echo ""
echo "Verify the format. Expected: one JSON object per line with a 'question' field."
echo "If the field name differs (e.g., 'text' or 'prompt'), update the loading code in train.py."
