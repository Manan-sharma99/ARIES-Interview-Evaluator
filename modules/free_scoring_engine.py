"""
Enhanced Scoring Engine for ARIES Interview System
Compatible with FREE AI Evaluator - No API costs!
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class ScoringEngine:
    """
    Calculates final interview scores by combining:
    1. AI Relevance Score (40%) - from Free AI evaluator
    2. Speech Emotion Score (25%) - from ML model
    3. Fluency Score (20%) - from filler word analysis
    4. Communication Quality (15%) - from sentiment analysis
    
    100% FREE - Works with free_ai_evaluator.py
    """
    
    def __init__(self):
        """Initialize scoring weights and emotion mappings"""
        
        # Weights for final score calculation
        self.weights = {
            'relevance': 0.40,      # AI-powered answer quality
            'emotion': 0.25,        # Speech emotion analysis
            'fluency': 0.20,        # Filler words and pace
            'communication': 0.15   # Overall sentiment
        }
        
        # Emotion to score mapping
        self.emotion_scores = {
            'confident': 90,
            'neutral': 65,
            'nervous': 40,
            'stressed': 30
        }
        
        # Grade boundaries
        self.grade_boundaries = {
            'A+': 95,
            'A': 85,
            'B+': 75,
            'B': 70,
            'C+': 60,
            'C': 55,
            'D': 40,
            'F': 0
        }
    
    def calculate_final_score(
        self,
        ai_relevance_score: float,
        emotion: str,
        filler_count: int,
        answer_length_words: int,
        sentiment_score: Optional[float] = None
    ) -> Dict:
        """
        Calculate final interview score
        
        Args:
            ai_relevance_score: 0-100 score from FREE AI evaluator
            emotion: Detected emotion ('confident', 'neutral', 'nervous', 'stressed')
            filler_count: Number of filler words detected
            answer_length_words: Total words in answer
            sentiment_score: Optional sentiment score (0-100)
        
        Returns:
            Dictionary with scores, grade, and breakdown
        """
        
        # 1. AI Relevance Score (0-100, already calculated)
        relevance_score = max(0, min(100, ai_relevance_score))
        
        # 2. Emotion Score (convert emotion to numeric)
        emotion_score = self.emotion_scores.get(emotion.lower(), 50)
        
        # 3. Fluency Score (penalize filler words)
        fluency_score = self._calculate_fluency_score(filler_count, answer_length_words)
        
        # 4. Communication Quality Score
        if sentiment_score is not None:
            communication_score = max(0, min(100, sentiment_score))
        else:
            # Fallback: estimate from emotion
            communication_score = emotion_score
        
        # Calculate weighted final score
        final_score = (
            relevance_score * self.weights['relevance'] +
            emotion_score * self.weights['emotion'] +
            fluency_score * self.weights['fluency'] +
            communication_score * self.weights['communication']
        )
        
        # Round to 1 decimal place
        final_score = round(final_score, 1)
        
        # Determine grade
        grade = self._get_grade(final_score)
        
        # Calculate component percentages
        components = {
            'relevance': round(relevance_score, 1),
            'emotion': round(emotion_score, 1),
            'fluency': round(fluency_score, 1),
            'communication': round(communication_score, 1)
        }
        
        return {
            'final_score': final_score,
            'grade': grade,
            'components': components,
            'weights': self.weights,
            'emotion_detected': emotion,
            'filler_words': filler_count,
            'assessment': self._get_assessment(final_score, grade)
        }
    
    def _calculate_fluency_score(self, filler_count: int, answer_length_words: int) -> float:
        """
        Calculate fluency score based on filler word ratio
        
        Score = 100 - (filler_ratio * 100)
        But with some leniency (occasional fillers are natural)
        """
        if answer_length_words == 0:
            return 0
        
        filler_ratio = filler_count / answer_length_words
        
        # Scoring thresholds
        if filler_ratio <= 0.02:  # ≤2% fillers: Excellent
            score = 100
        elif filler_ratio <= 0.05:  # 2-5%: Good
            score = 90 - (filler_ratio - 0.02) * 1000
        elif filler_ratio <= 0.10:  # 5-10%: Fair
            score = 70 - (filler_ratio - 0.05) * 800
        elif filler_ratio <= 0.15:  # 10-15%: Poor
            score = 50 - (filler_ratio - 0.10) * 600
        else:  # >15%: Very Poor
            score = max(20, 50 - (filler_ratio - 0.15) * 400)
        
        return round(max(0, min(100, score)), 1)
    
    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        for grade, threshold in self.grade_boundaries.items():
            if score >= threshold:
                return grade
        return 'F'
    
    def _get_assessment(self, score: float, grade: str) -> str:
        """Provide qualitative assessment based on score"""
        if score >= 85:
            return "Outstanding performance! You demonstrated excellent answer quality, strong confidence, and clear communication."
        elif score >= 70:
            return "Good performance. Your answer was relevant and well-delivered with minor areas for improvement."
        elif score >= 55:
            return "Average performance. Focus on improving answer depth and reducing filler words."
        elif score >= 40:
            return "Below average. Work on answer relevance, confidence, and fluency before your next interview."
        else:
            return "Needs significant improvement. Practice structuring answers and speaking more confidently."
    
    def calculate_session_metrics(self, session_scores: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics across multiple answers in a session
        
        Args:
            session_scores: List of score dictionaries from calculate_final_score()
        
        Returns:
            Session-level metrics including averages and ESI
        """
        if not session_scores:
            return {}
        
        # Extract all scores
        final_scores = [s['final_score'] for s in session_scores]
        emotion_scores = [s['components']['emotion'] for s in session_scores]
        relevance_scores = [s['components']['relevance'] for s in session_scores]
        
        # Calculate averages
        avg_final_score = round(np.mean(final_scores), 1)
        avg_emotion_score = round(np.mean(emotion_scores), 1)
        avg_relevance_score = round(np.mean(relevance_scores), 1)
        
        # Calculate Emotion Stability Index (ESI)
        # ESI = 1 - (std_dev / 100) -> Higher is better (more consistent)
        emotion_std = np.std(emotion_scores)
        esi = round(max(0, 1 - (emotion_std / 100)), 3)
        
        # Count emotion distribution
        emotions = [s['emotion_detected'] for s in session_scores]
        emotion_counts = {
            'confident': emotions.count('confident'),
            'neutral': emotions.count('neutral'),
            'nervous': emotions.count('nervous'),
            'stressed': emotions.count('stressed')
        }
        
        # Determine overall grade
        overall_grade = self._get_grade(avg_final_score)
        
        return {
            'total_questions': len(session_scores),
            'average_score': avg_final_score,
            'overall_grade': overall_grade,
            'average_relevance': avg_relevance_score,
            'average_emotion': avg_emotion_score,
            'emotion_stability_index': esi,
            'emotion_distribution': emotion_counts,
            'highest_score': max(final_scores),
            'lowest_score': min(final_scores),
            'score_range': round(max(final_scores) - min(final_scores), 1),
            'assessment': self._get_assessment(avg_final_score, overall_grade)
        }
    
    def compare_sessions(self, session1: Dict, session2: Dict) -> Dict:
        """
        Compare two sessions to show improvement/decline
        
        Args:
            session1: Older session metrics
            session2: Newer session metrics
        
        Returns:
            Comparison metrics showing changes
        """
        score_change = round(session2['average_score'] - session1['average_score'], 1)
        relevance_change = round(session2['average_relevance'] - session1['average_relevance'], 1)
        esi_change = round(session2['emotion_stability_index'] - session1['emotion_stability_index'], 3)
        
        return {
            'score_change': score_change,
            'relevance_change': relevance_change,
            'esi_change': esi_change,
            'improved': score_change > 0,
            'improvement_percentage': round((score_change / session1['average_score']) * 100, 1) if session1['average_score'] > 0 else 0
        }


# Testing and demonstration
if __name__ == "__main__":
    print("📊 FREE Scoring Engine - Testing")
    print("=" * 70)
    
    scorer = ScoringEngine()
    
    # Test Case 1: Excellent answer
    print("\nTEST 1: Excellent Performance")
    print("-" * 70)
    score1 = scorer.calculate_final_score(
        ai_relevance_score=92,
        emotion='confident',
        filler_count=2,
        answer_length_words=150,
        sentiment_score=88
    )
    
    print(f"AI Relevance: {score1['components']['relevance']}/100")
    print(f"Emotion ({score1['emotion_detected']}): {score1['components']['emotion']}/100")
    print(f"Fluency: {score1['components']['fluency']}/100")
    print(f"Communication: {score1['components']['communication']}/100")
    print(f"\n🎯 Final Score: {score1['final_score']}/100 (Grade: {score1['grade']})")
    print(f"📝 {score1['assessment']}")
    
    # Test Case 2: Weak answer
    print("\n\nTEST 2: Weak Performance")
    print("-" * 70)
    score2 = scorer.calculate_final_score(
        ai_relevance_score=45,
        emotion='nervous',
        filler_count=18,
        answer_length_words=80,
        sentiment_score=42
    )
    
    print(f"AI Relevance: {score2['components']['relevance']}/100")
    print(f"Emotion ({score2['emotion_detected']}): {score2['components']['emotion']}/100")
    print(f"Fluency: {score2['components']['fluency']}/100")
    print(f"Communication: {score2['components']['communication']}/100")
    print(f"\n🎯 Final Score: {score2['final_score']}/100 (Grade: {score2['grade']})")
    print(f"📝 {score2['assessment']}")
    
    # Test Case 3: Session metrics
    print("\n\nTEST 3: Session Metrics (5 questions)")
    print("-" * 70)
    
    session_scores = [
        scorer.calculate_final_score(85, 'confident', 3, 120, 82),
        scorer.calculate_final_score(78, 'neutral', 5, 100, 75),
        scorer.calculate_final_score(92, 'confident', 1, 140, 90),
        scorer.calculate_final_score(68, 'neutral', 8, 90, 70),
        scorer.calculate_final_score(88, 'confident', 2, 130, 85),
    ]
    
    session_metrics = scorer.calculate_session_metrics(session_scores)
    
    print(f"Total Questions: {session_metrics['total_questions']}")
    print(f"Average Score: {session_metrics['average_score']}/100")
    print(f"Overall Grade: {session_metrics['overall_grade']}")
    print(f"Average Relevance: {session_metrics['average_relevance']}/100")
    print(f"Emotion Stability Index (ESI): {session_metrics['emotion_stability_index']}")
    print(f"\nEmotion Distribution:")
    for emotion, count in session_metrics['emotion_distribution'].items():
        print(f"  {emotion}: {count}")
    print(f"\n📝 {session_metrics['assessment']}")
    
    print("\n" + "=" * 70)
    print("✅ Scoring Engine testing complete!")
    print("💰 Cost: $0.00 - Completely FREE!")
    print("\nWeight Distribution:")
    for component, weight in scorer.weights.items():
        print(f"  {component.capitalize()}: {weight*100}%")
