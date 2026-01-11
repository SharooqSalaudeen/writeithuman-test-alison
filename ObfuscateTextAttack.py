"""
Hybrid Obfuscation using ALISON's Integrated Gradients + TextAttack
Combines IG pattern identification with TextAttack's constrained word replacement
"""

from Utils import *
from NN import *
import torch
import numpy as np
from captum.attr import IntegratedGradients
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from sentence_transformers import SentenceTransformer, util
import argparse


class AuthorshipModelWrapper:
    """Wrapper for ALISON model to work with TextAttack constraints"""

    def __init__(self, model, features, scaler, device):
        self.model = model
        self.features = features
        self.scaler = scaler
        self.device = device
        self.model.eval()

    def __call__(self, text_list):
        """Predict author probabilities for a list of texts"""
        if isinstance(text_list, str):
            text_list = [text_list]

        # Extract features for each text
        feature_vectors = []
        for text in text_list:
            # Get POS tagged text
            pos_text = tag([text])[0]
            # Generate feature vector
            feature_vec = ngram_rep(text, pos_text, self.features)
            feature_vectors.append(feature_vec)

        # Scale features
        X = self.scaler.transform(np.array(feature_vectors))
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def predict_author(self, text):
        """Get predicted author for a single text"""
        probs = self(text)
        return np.argmax(probs[0])


def extract_revealing_patterns(text, label, model_wrapper, features, num_char, num_pos,
                               L=15, c=1.35, min_length=1, ig_steps=1024):
    """
    Use Integrated Gradients to identify top-L most revealing POS patterns
    Returns: List of (start_idx, end_idx, pattern_importance) tuples
    """
    print('  Analyzing revealing patterns with Integrated Gradients...')

    # Get POS tagged text
    pos_text = tag([text])[0]

    # Generate feature vector
    feature_vec = ngram_rep(text, pos_text, features)
    X = model_wrapper.scaler.transform(np.array([feature_vec]))
    X_tensor = torch.FloatTensor(X).to(model_wrapper.device)

    # Compute attributions
    ig = IntegratedGradients(model_wrapper.model)
    label_tensor = torch.tensor(
        [label], dtype=torch.int64).to(model_wrapper.device)

    with torch.no_grad():
        attributions = ig.attribute(
            X_tensor, target=label_tensor, n_steps=ig_steps)
        attributions = attributions.cpu().numpy()[0]

    # Apply length scaling
    mult = [c ** len(feature) for feature in features]
    attributions = np.multiply(attributions, mult)

    # Filter POS features only
    def isValid(index):
        return index >= num_char and index < num_char + num_pos

    # Rank by importance
    ranked_indexes = np.argsort(attributions)[::-1]
    ranked_indexes = [idx for idx in ranked_indexes if isValid(idx)]

    # Get top-L patterns
    top_patterns = [features[idx]
                    for idx in ranked_indexes[:L*3]]  # Get more candidates
    top_patterns = [p for p in top_patterns if len(p) >= min_length][:L]

    print(f'  Found {len(top_patterns)} revealing patterns')
    return top_patterns


def find_pattern_intervals(text, patterns):
    """
    Find word intervals in text that match the revealing POS patterns
    Returns: List of (start_word_idx, end_word_idx) tuples
    """
    words = tokenize(text)

    # Get POS tags for each word
    doc = nlp(' '.join(words))
    pos_tags = [tags[token.pos_]
                if token.pos_ in tags else token.pos_ for token in doc]

    intervals = []
    pattern_map = {}  # Track which pattern each interval comes from

    for pattern in patterns:
        # Find all occurrences of this pattern
        pattern_len = len(pattern)
        for i in range(len(pos_tags) - pattern_len + 1):
            if ''.join(pos_tags[i:i+pattern_len]) == pattern:
                interval = (i, i + pattern_len)
                intervals.append(interval)
                pattern_map[interval] = pattern

    # Remove overlapping intervals, keep highest priority
    intervals = sorted(set(intervals))
    filtered = []
    for interval in intervals:
        # Check if it overlaps with any already selected
        overlaps = False
        for sel_start, sel_end in filtered:
            if not (interval[1] <= sel_start or interval[0] >= sel_end):
                overlaps = True
                break
        if not overlaps:
            filtered.append(interval)

    print(f'  Identified {len(filtered)} word intervals to modify')
    return filtered


def replace_with_textattack(text, intervals, model_wrapper, original_label,
                            min_similarity=0.85, max_candidates=50):
    """
    Use TextAttack to replace words in specified intervals with semantic constraints
    """
    print(
        f'  Applying TextAttack word replacement (similarity >= {min_similarity})...')

    # Initialize transformations and constraints
    transformation = WordSwapMaskedLM(
        method="bert-base-uncased", max_candidates=max_candidates)

    # Semantic similarity constraint using sentence transformers
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    original_embedding = similarity_model.encode(text, convert_to_tensor=True)

    words = tokenize(text)
    modified_words = words.copy()
    modifications_made = 0

    # Process each interval
    for start_idx, end_idx in intervals:
        interval_words = words[start_idx:end_idx]

        # Try to replace each word in the interval
        for word_idx in range(start_idx, end_idx):
            if word_idx >= len(modified_words):
                continue

            original_word = modified_words[word_idx]

            # Skip if stopword or punctuation
            if original_word.lower() in stopwords.words('english') or original_word in punctuation:
                continue

            # Get replacement candidates
            temp_text = ' '.join(modified_words)
            try:
                # Use BERT to suggest replacements
                masked_text = modified_words.copy()
                masked_text[word_idx] = '[MASK]'
                masked_input = ' '.join(masked_text)

                # Get BERT predictions
                from transformers import pipeline
                if not hasattr(replace_with_textattack, 'unmasker'):
                    replace_with_textattack.unmasker = pipeline(
                        'fill-mask',
                        model='bert-base-uncased',
                        device=0 if torch.cuda.is_available() else -1
                    )

                predictions = replace_with_textattack.unmasker(
                    masked_input, top_k=max_candidates)

                # Test each candidate
                best_replacement = None
                best_score = -1

                for pred in predictions:
                    candidate_word = pred['token_str'].strip()

                    # Skip if same as original
                    if candidate_word.lower() == original_word.lower():
                        continue

                    # Create candidate text
                    candidate_words = modified_words.copy()
                    candidate_words[word_idx] = candidate_word
                    candidate_text = ' '.join(candidate_words)

                    # Check semantic similarity
                    candidate_embedding = similarity_model.encode(
                        candidate_text, convert_to_tensor=True)
                    similarity = util.cos_sim(
                        original_embedding, candidate_embedding).item()

                    if similarity < min_similarity:
                        continue

                    # Check if it fools the classifier
                    predicted_author = model_wrapper.predict_author(
                        candidate_text)

                    # Score: prefer changes that fool classifier + high similarity
                    fool_score = 1.0 if predicted_author != original_label else 0.0
                    score = fool_score * 2 + similarity

                    if score > best_score:
                        best_score = score
                        best_replacement = candidate_word

                # Apply best replacement if found
                if best_replacement:
                    modified_words[word_idx] = best_replacement
                    modifications_made += 1
                    print(
                        f'    Replaced "{original_word}" → "{best_replacement}" (sim: {similarity:.3f})')

            except Exception as e:
                print(f'    Warning: Could not replace "{original_word}": {e}')
                continue

    modified_text = ' '.join(modified_words)

    # Final similarity check
    final_embedding = similarity_model.encode(
        modified_text, convert_to_tensor=True)
    final_similarity = util.cos_sim(original_embedding, final_embedding).item()

    print(
        f'  Made {modifications_made} modifications, final similarity: {final_similarity:.3f}')

    return modified_text, final_similarity


def obfuscate_text_hybrid(text, model_wrapper, features, num_char, num_pos,
                          original_label=0, L=15, c=1.35, min_length=1,
                          ig_steps=1024, min_similarity=0.85):
    """
    Hybrid obfuscation: IG pattern identification + TextAttack replacement
    """
    print(f'\nObfuscating text ({len(text)} characters)...')

    # Step 1: Use IG to identify revealing patterns
    patterns = extract_revealing_patterns(
        text, original_label, model_wrapper, features,
        num_char, num_pos, L, c, min_length, ig_steps
    )

    if not patterns:
        print('  No revealing patterns found.')
        return text, 1.0

    # Step 2: Find word intervals matching these patterns
    intervals = find_pattern_intervals(text, patterns)

    if not intervals:
        print('  No matching intervals found in text.')
        return text, 1.0

    # Step 3: Use TextAttack to replace words in intervals
    obfuscated_text, similarity = replace_with_textattack(
        text, intervals, model_wrapper, original_label, min_similarity
    )

    # Verify obfuscation success
    original_pred = model_wrapper.predict_author(text)
    obfuscated_pred = model_wrapper.predict_author(obfuscated_text)

    success = "✓" if original_pred != obfuscated_pred else "✗"
    print(f'\n{success} Original author: {original_pred} → Obfuscated author: {obfuscated_pred}')
    print(f'  Semantic similarity: {similarity:.3f}')

    return obfuscated_text, similarity


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid ALISON + TextAttack Obfuscation')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text', '-t', help='Direct text input for obfuscation')
    input_group.add_argument(
        '--file', '-f', help='Path to file containing texts')

    # Model options
    parser.add_argument('--dir', '-d', required=True,
                        help='Path to trained model directory')
    parser.add_argument('--output', '-o', default='obfuscated_output.txt',
                        help='Output file path')

    # Obfuscation parameters
    parser.add_argument('--L', type=int, default=15,
                        help='Number of top POS patterns to target')
    parser.add_argument('--c', type=float, default=1.35,
                        help='Length scaling constant')
    parser.add_argument('--min_length', type=int, default=1,
                        help='Minimum pattern length')
    parser.add_argument('--ig_steps', type=int, default=1024,
                        help='Integrated Gradients steps')
    parser.add_argument('--min_similarity', type=float, default=0.85,
                        help='Minimum semantic similarity (0-1)')
    parser.add_argument('--max_candidates', type=int, default=50,
                        help='Max replacement candidates per word')

    args = parser.parse_args()

    print('='*70)
    print('HYBRID OBFUSCATION: ALISON IG + TextAttack')
    print('='*70)

    # Load model and features
    print('\nLoading model and features...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    features = pickle.load(open(os.path.join(args.dir, 'features.pkl'), 'rb'))
    features_flat = np.array(features).flatten().tolist()

    # Load or create scaler
    scaler_path = os.path.join(args.dir, 'Scaler.pkl')
    if os.path.exists(scaler_path):
        Scaler = pickle.load(open(scaler_path, 'rb'))
    else:
        X_train = pickle.load(
            open(os.path.join(args.dir, 'X_train.pkl'), 'rb'))
        Scaler = sklearn.preprocessing.StandardScaler()
        Scaler.fit(X_train)

    # Load model with correct architecture
    X_train = pickle.load(open(os.path.join(args.dir, 'X_train.pkl'), 'rb'))
    y_train = pickle.load(open(os.path.join(args.dir, 'y_train.pkl'), 'rb'))

    input_size = X_train.shape[1]
    num_authors = len(np.unique(y_train))

    model = Model(input_size, num_authors)
    model.load_state_dict(torch.load(os.path.join(
        args.dir, 'model.pt'), map_location=device))
    model.to(device)
    model.eval()

    print(f'Model loaded: {input_size} features, {num_authors} authors')

    # Create model wrapper
    model_wrapper = AuthorshipModelWrapper(
        model, features_flat, Scaler, device)

    num_char = features[0].size
    num_pos = features[1].size

    # Load input text
    if args.text:
        texts = [args.text]
        labels = [0]  # Assume label 0 for single text
    else:
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.partition(' ') for line in f.readlines()]
            texts = [line[2].strip() for line in lines]
            labels = [int(line[0]) for line in lines]

    print(f'\nProcessing {len(texts)} text(s)...')

    # Obfuscate each text
    results = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        print(f'\n{"="*70}')
        print(f'Text {i+1}/{len(texts)}')
        print(f'{"="*70}')

        obfuscated, similarity = obfuscate_text_hybrid(
            text, model_wrapper, features_flat, num_char, num_pos,
            original_label=label,
            L=args.L,
            c=args.c,
            min_length=args.min_length,
            ig_steps=args.ig_steps,
            min_similarity=args.min_similarity
        )

        results.append({
            'original': text,
            'obfuscated': obfuscated,
            'similarity': similarity
        })

    # Save results
    print(f'\n{"="*70}')
    print('RESULTS')
    print('='*70)

    with open(args.output, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            f.write(f'Text {i+1}:\n')
            f.write(f'Original:\n{result["original"]}\n\n')
            f.write(f'Obfuscated:\n{result["obfuscated"]}\n\n')
            f.write(f'Similarity: {result["similarity"]:.3f}\n')
            f.write('-'*70 + '\n\n')

    print(f'\nResults saved to: {args.output}')
    print(
        f'Average similarity: {np.mean([r["similarity"] for r in results]):.3f}')
    print('\nDone!')


if __name__ == "__main__":
    main()
