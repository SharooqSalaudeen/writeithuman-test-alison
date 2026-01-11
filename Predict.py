from Utils import *
from NN import *
import argparse


def predict_author(text, model_dir, authors_total):
    """Predict the author of a given text"""

    print('Loading model and features...')

    # Load saved features
    features = pickle.load(open(os.path.join(model_dir, 'features.pkl'), 'rb'))

    # Load or create scaler
    scaler_path = os.path.join(model_dir, 'Scaler.pkl')
    if os.path.exists(scaler_path):
        Scaler = pickle.load(open(scaler_path, 'rb'))
        X_train = None
    else:
        # Create scaler from saved training data
        print('Scaler.pkl not found, creating from X_train.pkl...')
        X_train = pickle.load(
            open(os.path.join(model_dir, 'X_train.pkl'), 'rb'))
        Scaler = sklearn.preprocessing.StandardScaler()
        Scaler.fit(X_train)

    # Calculate correct input size from actual training data
    if X_train is None:
        X_train = pickle.load(
            open(os.path.join(model_dir, 'X_train.pkl'), 'rb'))
    input_size = X_train.shape[1]

    # Get actual number of authors from model checkpoint
    checkpoint = torch.load(os.path.join(
        model_dir, 'model.pt'), map_location=device)
    # The output layer is 'stack.18.weight' with shape [num_authors, 128]
    actual_authors = checkpoint['stack.18.weight'].shape[0]

    print(f'Model info: {input_size} features, {actual_authors} authors')

    # Load model
    model = Model(input_size, actual_authors)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    print('Processing text...')

    # Tag the text
    pos_text = tag([text])[0]

    # Extract features
    X = ngram_rep(text, pos_text, features)
    X = np.array([X])
    X = Scaler.transform(X)

    # Predict
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        output = model(X_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_author = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_author].item()

    return predicted_author, confidence, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Predict the author of a text')

    parser.add_argument('--text', '-t', type=str,
                        help='Text to analyze (quote in "...")')
    parser.add_argument('--file', '-f', type=str,
                        help='Path to text file to analyze')
    parser.add_argument('--dir', '-d', required=True,
                        help='Path to trained model directory')
    parser.add_argument('--authors_total', '-at', type=int,
                        required=True, help='Number of authors')

    args = parser.parse_args()

    # Get text from argument or file
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    else:
        print("Error: Please provide either --text or --file")
        return

    print('\n' + '='*60)
    print('AUTHOR ATTRIBUTION')
    print('='*60)
    print(f'\nText (first 200 chars): {text[:200]}...\n')

    # Predict
    predicted_author, confidence, all_probs = predict_author(
        text, args.dir, args.authors_total)

    print('='*60)
    print(f'PREDICTED AUTHOR: {predicted_author}')
    print(f'CONFIDENCE: {confidence*100:.2f}%')
    print('='*60)

    # Show all probabilities
    print('\nAll Author Probabilities:')
    print('-'*60)
    for i, prob in enumerate(all_probs):
        bar = '█' * int(prob * 50)
        print(f'Author {i}: {prob*100:6.2f}% {bar}')
    print('-'*60)


if __name__ == "__main__":
    main()
