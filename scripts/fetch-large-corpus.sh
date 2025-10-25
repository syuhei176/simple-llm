#!/bin/bash
# Fetch multiple public domain books from Project Gutenberg to build a large corpus

echo "=== Fetching Large Corpus from Project Gutenberg ==="
echo ""

# Create data/corpus directory
mkdir -p data/corpus

# Project Gutenberg books (all public domain)
BOOKS=(
  "https://www.gutenberg.org/files/1342/1342-0.txt:pride-and-prejudice.txt"
  "https://www.gutenberg.org/files/11/11-0.txt:alice-in-wonderland.txt"
  "https://www.gutenberg.org/files/1661/1661-0.txt:sherlock-holmes.txt"
  "https://www.gutenberg.org/files/98/98-0.txt:tale-of-two-cities.txt"
  "https://www.gutenberg.org/files/84/84-0.txt:frankenstein.txt"
)

# Download each book
for entry in "${BOOKS[@]}"; do
  IFS=':' read -r url filename <<< "$entry"
  echo "Downloading: $filename"
  npx ts-node --project tsconfig.scripts.json scripts/fetch-text.ts "$url" "data/corpus/$filename"
  echo ""
done

# Combine all books into one large corpus
echo "Combining all books into large-corpus.txt..."
cat data/corpus/*.txt > data/large-corpus.txt

echo ""
echo "✓ Large corpus created successfully!"
echo "  Location: data/large-corpus.txt"
wc -w data/large-corpus.txt
