from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from modules.questions_bank import IDEAL_ANSWERS

print("Loading NLP model... (first time may take 1-2 minutes)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("NLP model ready!")

CONFIDENT_PHRASES = ['i am', 'i have', 'i can', 'i will', 'i believe',
                     'i know', 'definitely', 'certainly', 'absolutely']
WEAK_PHRASES = ['i think maybe', 'i guess', 'i am not sure', 'probably',
                'i might', 'sort of', 'kind of', 'i hope']

def evaluate_relevance(question, user_answer):
    if not user_answer or len(user_answer.strip()) < 5:
        return 0.0
    ideal_answer = IDEAL_ANSWERS.get(question, question)
    embeddings = model.encode([user_answer, ideal_answer])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(similarity) * 100, 2)

def analyze_sentiment(text):
    text_lower = text.lower()
    positive_words = ['good', 'great', 'excellent', 'strong', 'passionate',
                      'excited', 'confident', 'successful', 'achieve', 'love',
                      'enjoy', 'dedicated', 'motivated', 'proud', 'happy']
    negative_words = ['bad', 'failed', 'struggle', 'difficult', 'hate',
                      'nervous', 'scared', 'worried', 'problem', 'issue', 'weak']
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    if pos_count > neg_count:
        return "Positive", min(100, 60 + (pos_count * 5))
    elif neg_count > pos_count:
        return "Negative", max(0, 40 - (neg_count * 5))
    return "Neutral", 50

def analyze_confidence_language(text):
    text_lower = text.lower()
    confident_count = sum(1 for phrase in CONFIDENT_PHRASES if phrase in text_lower)
    weak_count = sum(1 for phrase in WEAK_PHRASES if phrase in text_lower)
    return min(100, max(0, 60 + (confident_count * 8) - (weak_count * 10)))

def evaluate_answer(question, user_answer):
    relevance_score = evaluate_relevance(question, user_answer)
    sentiment, sentiment_score = analyze_sentiment(user_answer)
    confidence_score = analyze_confidence_language(user_answer)
    word_count = len(user_answer.split())
    length_penalty = 0.5 if word_count < 10 else 0.9 if word_count > 200 else 1.0
    relevance_score = round(relevance_score * length_penalty, 2)
    return {
        'relevance_score': relevance_score,
        'sentiment': sentiment,
        'sentiment_score': sentiment_score,
        'confidence_score': confidence_score,
        'word_count': word_count
    }

if __name__ == "__main__":
    question = "Tell me about yourself"
    answer = "I am a computer science student with a passion for AI and machine learning. I have worked on several projects and I am confident in my ability to learn quickly and contribute to a team."
    print(f"Question: {question}")
    results = evaluate_answer(question, answer)
    print(f"Relevance Score:   {results['relevance_score']}/100")
    print(f"Sentiment:         {results['sentiment']} ({results['sentiment_score']}/100)")
    print(f"Confidence Score:  {results['confidence_score']}/100")
    print(f"Word Count:        {results['word_count']} words")
    print(f"\nTotal questions in bank: {sum(len(v) for v in __import__('modules.questions_bank', fromlist=['QUESTIONS']).QUESTIONS.values())}")
