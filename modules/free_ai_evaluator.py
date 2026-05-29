"""
FREE AI Evaluator Module for ARIES Interview System
Uses local Hugging Face models - NO API KEYS NEEDED!
Works 100% offline after initial model download
"""

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from typing import Dict, List, Optional
import re

class FreeAIEvaluator:
    """
    Free AI-powered answer evaluator using local models
    - Sentence-BERT for semantic similarity
    - DistilBERT for sentiment analysis
    - BART for text quality assessment
    - Custom rule-based analysis for STAR method and key points
    
    100% FREE - No API costs!
    """
    
    def __init__(self):
        """
        Initialize all free models (downloads on first run, then cached)
        Models are small and fast: ~500MB total
        """
        print("🤖 Loading free AI models (first time may take a few minutes)...")
        
        # 1. Sentence-BERT for semantic similarity (133MB)
        print("   Loading Sentence-BERT...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Sentiment analyzer for answer tone (255MB)
        print("   Loading sentiment analyzer...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU mode (use 0 for GPU if available)
        )
        
        # 3. Zero-shot classifier for category detection (892MB - optional, can disable)
        # Commenting out to save memory - we already have categories from user
        # print("   Loading zero-shot classifier...")
        # self.classifier = pipeline("zero-shot-classification", 
        #                           model="facebook/bart-large-mnli",
        #                           device=-1)
        
        print("✅ All models loaded successfully!\n")
        
        # STAR method keywords
        self.star_keywords = {
            'situation': ['situation', 'scenario', 'context', 'background', 'when', 'where'],
            'task': ['task', 'challenge', 'problem', 'goal', 'objective', 'responsibility'],
            'action': ['action', 'did', 'implemented', 'created', 'developed', 'organized', 'led'],
            'result': ['result', 'outcome', 'achieved', 'improved', 'increased', 'decreased', 'successfully']
        }
        
        # Filler words for fluency check
        self.filler_words = [
            'um', 'uh', 'like', 'you know', 'basically', 'actually', 
            'literally', 'sort of', 'kind of', 'i mean', 'so yeah',
            'well', 'hmm', 'er', 'ah', 'okay so', 'right', 'yeah so'
        ]
    
    def evaluate_answer(
        self,
        question: str,
        student_answer: str,
        ideal_answer: Optional[str] = None,
        category: str = "General"
    ) -> Dict:
        """
        Evaluate answer using multiple free AI techniques
        
        Returns:
            Dictionary with relevance_score, feedback, strengths, improvements
        """
        
        # Clean inputs
        question = question.strip()
        student_answer = student_answer.strip()
        
        if not student_answer or len(student_answer) < 10:
            return self._empty_answer_response()
        
        # Calculate multiple scores
        scores = {
            'semantic_similarity': self._calculate_semantic_similarity(question, student_answer, ideal_answer),
            'answer_completeness': self._assess_completeness(student_answer),
            'structure_quality': self._assess_structure(student_answer, category),
            'sentiment_score': self._analyze_sentiment(student_answer),
            'keyword_coverage': self._check_keyword_coverage(question, student_answer)
        }
        
        # Calculate final relevance score (weighted average)
        relevance_score = self._calculate_weighted_score(scores, ideal_answer)
        
        # Generate detailed feedback
        feedback_components = self._generate_feedback(student_answer, category, scores)
        
        return {
            'relevance_score': round(relevance_score, 1),
            'feedback': feedback_components['feedback'],
            'strengths': feedback_components['strengths'],
            'improvements': feedback_components['improvements'],
            'key_points_covered': feedback_components['covered'],
            'key_points_missed': feedback_components['missed'],
            'detailed_scores': scores
        }
    
    def _calculate_semantic_similarity(
        self,
        question: str,
        student_answer: str,
        ideal_answer: Optional[str]
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings
        Returns score 0-100
        """
        # Encode texts
        question_embedding = self.sentence_model.encode(question)
        answer_embedding = self.sentence_model.encode(student_answer)
        
        # Calculate cosine similarity with question
        question_similarity = np.dot(question_embedding, answer_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(answer_embedding)
        )
        
        # If ideal answer provided, also compare with that
        if ideal_answer:
            ideal_embedding = self.sentence_model.encode(ideal_answer)
            ideal_similarity = np.dot(ideal_embedding, answer_embedding) / (
                np.linalg.norm(ideal_embedding) * np.linalg.norm(answer_embedding)
            )
            # Average both similarities
            similarity = (question_similarity * 0.4 + ideal_similarity * 0.6)
        else:
            similarity = question_similarity
        
        # Convert to 0-100 scale
        return max(0, min(100, similarity * 100))
    
    def _assess_completeness(self, answer: str) -> float:
        """
        Assess answer completeness based on length and detail
        Returns score 0-100
        """
        words = answer.split()
        word_count = len(words)
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        
        # Ideal answer length: 80-200 words
        if word_count < 30:
            length_score = word_count / 30 * 60  # Too short
        elif word_count <= 200:
            length_score = 100  # Ideal range
        else:
            length_score = max(70, 100 - (word_count - 200) / 10)  # Too long
        
        # Sentence variety (not all super short or super long)
        avg_words_per_sentence = word_count / max(1, sentence_count)
        if 10 <= avg_words_per_sentence <= 20:
            structure_score = 100
        else:
            structure_score = 70
        
        return (length_score * 0.7 + structure_score * 0.3)
    
    def _assess_structure(self, answer: str, category: str) -> float:
        """
        Assess structure quality (STAR for behavioral, clarity for technical)
        Returns score 0-100
        """
        answer_lower = answer.lower()
        
        if category.lower() in ['behavioral', 'leadership', 'situational']:
            # Check for STAR method
            star_score = 0
            star_found = {}
            
            for component, keywords in self.star_keywords.items():
                found = any(keyword in answer_lower for keyword in keywords)
                star_found[component] = found
                if found:
                    star_score += 25  # Each component worth 25 points
            
            # Bonus for having all 4 components
            if star_score == 100:
                return 100
            elif star_score >= 75:
                return 85
            elif star_score >= 50:
                return 70
            else:
                return max(40, star_score)
        
        else:
            # For technical/general: check for examples, explanations
            has_example = any(word in answer_lower for word in ['example', 'for instance', 'such as', 'like when'])
            has_explanation = any(word in answer_lower for word in ['because', 'since', 'therefore', 'thus', 'this means'])
            
            score = 60  # Base score
            if has_example:
                score += 20
            if has_explanation:
                score += 20
            
            return score
    
    def _analyze_sentiment(self, answer: str) -> float:
        """
        Analyze sentiment/tone of the answer
        Returns score 0-100
        """
        try:
            # Analyze overall sentiment
            result = self.sentiment_analyzer(answer[:512])[0]  # Max 512 tokens
            
            if result['label'] == 'POSITIVE':
                # Positive tone is good (confidence)
                return result['score'] * 100
            else:
                # Negative tone (might indicate stress/nervousness)
                return (1 - result['score']) * 100
        except:
            return 70  # Default neutral score
    
    def _check_keyword_coverage(self, question: str, answer: str) -> float:
        """
        Check if important keywords from question appear in answer
        Returns score 0-100
        """
        # Extract important words from question (nouns, verbs)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'can', 'could', 'may', 'might', 'must', 'what', 'when', 'where', 
                     'who', 'why', 'how', 'tell', 'describe', 'explain', 'about', 'me', 'you'}
        
        important_question_words = question_words - stop_words
        
        if not important_question_words:
            return 70
        
        # Calculate overlap
        overlap = important_question_words.intersection(answer_words)
        coverage = len(overlap) / len(important_question_words)
        
        return min(100, coverage * 150)  # Boost score slightly
    
    def _calculate_weighted_score(self, scores: Dict, has_ideal_answer: bool) -> float:
        """
        Calculate final weighted relevance score
        """
        if has_ideal_answer:
            # With ideal answer: prioritize semantic similarity
            weights = {
                'semantic_similarity': 0.45,
                'answer_completeness': 0.20,
                'structure_quality': 0.20,
                'sentiment_score': 0.05,
                'keyword_coverage': 0.10
            }
        else:
            # Without ideal answer: more balanced
            weights = {
                'semantic_similarity': 0.30,
                'answer_completeness': 0.25,
                'structure_quality': 0.25,
                'sentiment_score': 0.05,
                'keyword_coverage': 0.15
            }
        
        final_score = sum(scores[key] * weight for key, weight in weights.items())
        return max(0, min(100, final_score))
    
    def _generate_feedback(self, answer: str, category: str, scores: Dict) -> Dict:
        """
        Generate detailed feedback based on scores
        """
        strengths = []
        improvements = []
        covered = []
        missed = []
        
        # Analyze strengths
        if scores['answer_completeness'] >= 80:
            strengths.append("Well-detailed answer with good length")
            covered.append("Comprehensive coverage of the topic")
        
        if scores['structure_quality'] >= 80:
            if category.lower() in ['behavioral', 'leadership']:
                strengths.append("Good use of STAR method (Situation, Task, Action, Result)")
                covered.append("Clear structure with specific examples")
            else:
                strengths.append("Well-structured response with examples")
                covered.append("Logical flow and explanations")
        
        if scores['sentiment_score'] >= 70:
            strengths.append("Confident and positive tone")
        
        if scores['keyword_coverage'] >= 70:
            strengths.append("Directly addresses the question asked")
            covered.append("Relevant keywords and concepts mentioned")
        
        # Identify improvements
        if scores['answer_completeness'] < 60:
            improvements.append("Provide more details and examples to strengthen your answer")
            missed.append("Insufficient depth - expand on key points")
        
        if scores['structure_quality'] < 60:
            if category.lower() in ['behavioral', 'leadership']:
                improvements.append("Follow the STAR method: clearly describe Situation, Task, Action, and Result")
                missed.append("Missing one or more STAR components")
            else:
                improvements.append("Improve structure by adding specific examples or explanations")
        
        if scores['sentiment_score'] < 50:
            improvements.append("Speak more confidently - avoid negative or uncertain language")
        
        if scores['keyword_coverage'] < 50:
            improvements.append("Make sure to directly address all parts of the question")
            missed.append("Some key concepts from the question not addressed")
        
        if scores['semantic_similarity'] < 60:
            improvements.append("Ensure your answer is more closely related to what was asked")
            missed.append("Answer may be going off-topic")
        
        # Generate overall feedback
        avg_score = scores['semantic_similarity']
        if avg_score >= 85:
            feedback = "Excellent answer! You demonstrated strong understanding and clear communication."
        elif avg_score >= 70:
            feedback = "Good answer with relevant content. Some areas could be strengthened."
        elif avg_score >= 55:
            feedback = "Adequate answer but needs improvement in depth and structure."
        else:
            feedback = "Answer needs significant improvement. Focus on relevance and completeness."
        
        return {
            'feedback': feedback,
            'strengths': strengths[:3],  # Top 3 strengths
            'improvements': improvements[:3],  # Top 3 improvements
            'covered': covered,
            'missed': missed
        }
    
    def _empty_answer_response(self) -> Dict:
        """Return response for empty/very short answers"""
        return {
            'relevance_score': 0,
            'feedback': "No answer provided or answer too short to evaluate.",
            'strengths': [],
            'improvements': ["Please provide a complete answer to the question"],
            'key_points_covered': [],
            'key_points_missed': ["Complete answer required"],
            'detailed_scores': {}
        }
    
    def get_quick_score(self, question: str, student_answer: str) -> float:
        """Quick score without detailed feedback"""
        result = self.evaluate_answer(question, student_answer)
        return result['relevance_score']


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("🆓 FREE AI EVALUATOR - No API Keys Needed!")
    print("=" * 80)
    
    # Initialize (downloads models on first run)
    evaluator = FreeAIEvaluator()
    
    # Test Case 1: Good behavioral answer
    print("\n" + "="*80)
    print("TEST 1: Behavioral Question - Good Answer")
    print("="*80)
    
    question1 = "Tell me about a time when you had to work under pressure."
    answer1 = """During my second year, I was leading a team project for our database course 
    while also preparing for mid-term exams. Two team members fell sick a week before the deadline. 
    I quickly reorganized tasks, took on the critical backend work myself, and held daily 15-minute 
    standups to keep everyone aligned. I also created a shared checklist so we could track progress. 
    We submitted on time and received an A grade. This taught me the importance of staying calm 
    and focusing on what I can control."""
    
    result1 = evaluator.evaluate_answer(question1, answer1, category="Behavioral")
    
    print(f"\n📊 Relevance Score: {result1['relevance_score']}/100")
    print(f"\n💬 Feedback: {result1['feedback']}")
    print(f"\n✅ Strengths:")
    for s in result1['strengths']:
        print(f"   • {s}")
    print(f"\n⚠️  Improvements:")
    for i in result1['improvements']:
        print(f"   • {i}")
    
    # Test Case 2: Weak technical answer
    print("\n" + "="*80)
    print("TEST 2: Technical Question - Weak Answer")
    print("="*80)
    
    question2 = "What is the difference between REST and GraphQL?"
    answer2 = "REST is an API and GraphQL is also an API. They are used for web development."
    
    result2 = evaluator.evaluate_answer(question2, answer2, category="Technical")
    
    print(f"\n📊 Relevance Score: {result2['relevance_score']}/100")
    print(f"\n💬 Feedback: {result2['feedback']}")
    print(f"\n⚠️  Improvements:")
    for i in result2['improvements']:
        print(f"   • {i}")
    print(f"\n❌ Key Points Missed:")
    for m in result2['key_points_missed']:
        print(f"   • {m}")
    
    print("\n" + "="*80)
    print("✅ FREE AI EVALUATOR READY TO USE!")
    print("💰 Cost: $0.00 - Everything runs locally!")
    print("="*80)
