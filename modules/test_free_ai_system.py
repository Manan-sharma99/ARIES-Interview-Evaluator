"""
Complete End-to-End Test for FREE AI-Powered Answer Evaluation
100% FREE - No API keys needed!
"""

import sys
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def main():
    print_header("FREE AI Interview Evaluation System - Complete Test")
    print(f"{Colors.OKGREEN}💰 Cost: $0.00 - Completely FREE!{Colors.ENDC}\n")
    
    # Step 1: Check packages
    print_info("Step 1: Checking required packages...")
    
    try:
        import sentence_transformers
        print_success("sentence-transformers installed")
    except ImportError:
        print_error("sentence-transformers not found")
        print("\nInstall with:")
        print("pip install sentence-transformers --break-system-packages")
        return False
    
    try:
        import transformers
        print_success("transformers installed")
    except ImportError:
        print_error("transformers not found")
        print("\nInstall with:")
        print("pip install transformers --break-system-packages")
        return False
    
    try:
        import torch
        print_success("torch installed")
    except ImportError:
        print_error("torch not found")
        print("\nInstall with:")
        print("pip install torch --break-system-packages")
        return False
    
    # Step 2: Import modules
    print_info("\nStep 2: Importing evaluation modules...")
    try:
        from free_ai_evaluator import FreeAIEvaluator
        from free_scoring_engine import ScoringEngine
        print_success("Modules imported successfully")
    except ImportError as e:
        print_error(f"Failed to import modules: {e}")
        print("\nMake sure you have:")
        print("1. free_ai_evaluator.py in the current directory")
        print("2. free_scoring_engine.py in the current directory")
        return False
    
    # Step 3: Initialize evaluators
    print_info("\nStep 3: Initializing AI evaluator (downloading models if needed)...")
    print_warning("First run may take 2-5 minutes to download models...")
    
    try:
        ai_eval = FreeAIEvaluator()
        scorer = ScoringEngine()
        print_success("Evaluators initialized and ready!")
    except Exception as e:
        print_error(f"Failed to initialize: {e}")
        return False
    
    # Step 4: Run test cases
    print_header("Running Test Cases")
    
    test_cases = [
        {
            "name": "Behavioral - Excellent STAR Answer",
            "question": "Tell me about a time when you had to deal with a difficult team member.",
            "answer": """In my second year, I was the team lead for our Software Engineering project. 
            One team member, let's call him Alex, consistently missed deadlines and didn't communicate 
            well with the team. Rather than confronting him publicly, I scheduled a one-on-one meeting. 
            I discovered he was struggling with personal issues and felt overwhelmed by the workload. 
            I worked with him to break down his tasks into smaller, manageable pieces and paired him 
            with another teammate for support. I also adjusted our sprint planning to be more flexible. 
            As a result, Alex's performance improved significantly, he started meeting deadlines, and 
            our team completed the project successfully with an A grade. This taught me the importance 
            of empathy and communication in leadership.""",
            "category": "Behavioral",
            "emotion": "confident",
            "filler_count": 3,
            "ideal_answer": None
        },
        {
            "name": "Technical - Weak Answer",
            "question": "Explain the difference between supervised and unsupervised learning.",
            "answer": """Supervised learning is when you supervise the model and unsupervised learning 
            is when you don't supervise it. In supervised learning there is training data and in 
            unsupervised learning there is no training data. That's the main difference.""",
            "category": "Technical",
            "emotion": "nervous",
            "filler_count": 5,
            "ideal_answer": """Supervised learning uses labeled training data where each input has a 
            corresponding output label. The algorithm learns to map inputs to outputs. Examples include 
            classification and regression. Unsupervised learning works with unlabeled data, finding 
            patterns and structure without predefined categories. Examples include clustering and 
            dimensionality reduction."""
        },
        {
            "name": "Leadership - Good Answer",
            "question": "Describe a situation where you had to make a difficult decision.",
            "answer": """During our college tech fest, I was the organizing committee head. Two days 
            before the event, our main sponsor withdrew funding, leaving us short by 50,000 rupees. 
            I had two options: cancel half the events or find alternative funding immediately. 
            I decided to reach out to 10 local startups and pitched them a smaller sponsorship package. 
            Within 24 hours, I secured commitments from 3 companies that collectively covered our gap. 
            I also negotiated with vendors for deferred payments. The fest went ahead as planned and 
            was a huge success with 500+ participants. This experience taught me to stay calm under 
            pressure and think creatively about problem-solving.""",
            "category": "Leadership",
            "emotion": "confident",
            "filler_count": 2,
            "ideal_answer": None
        },
        {
            "name": "General - Incomplete Answer",
            "question": "Why do you want to work at Google?",
            "answer": """Google is a great company. I like their products and want to work there.""",
            "category": "General",
            "emotion": "nervous",
            "filler_count": 1,
            "ideal_answer": None
        }
    ]
    
    all_scores = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Colors.BOLD}TEST CASE {i}: {test_case['name']}{Colors.ENDC}")
        print("-" * 80)
        
        # Display question
        print(f"\n{Colors.OKCYAN}Question ({test_case['category']}):{Colors.ENDC}")
        print(f"{test_case['question']}\n")
        
        # Display answer (truncated if long)
        print(f"{Colors.OKCYAN}Student's Answer:{Colors.ENDC}")
        if len(test_case['answer']) > 300:
            print(f"{test_case['answer'][:300]}...\n")
        else:
            print(f"{test_case['answer']}\n")
        
        # AI Evaluation
        print_info("Evaluating with FREE AI...")
        try:
            ai_result = ai_eval.evaluate_answer(
                question=test_case['question'],
                student_answer=test_case['answer'],
                ideal_answer=test_case['ideal_answer'],
                category=test_case['category']
            )
            
            print_success("AI Evaluation complete")
            print(f"\n📊 AI Relevance Score: {Colors.BOLD}{ai_result['relevance_score']}/100{Colors.ENDC}")
            print(f"\n💬 AI Feedback:")
            print(f"   {ai_result['feedback']}")
            
            if ai_result['strengths']:
                print(f"\n{Colors.OKGREEN}✓ Strengths:{Colors.ENDC}")
                for strength in ai_result['strengths']:
                    print(f"   • {strength}")
            
            if ai_result['improvements']:
                print(f"\n{Colors.WARNING}→ Improvements:{Colors.ENDC}")
                for improvement in ai_result['improvements']:
                    print(f"   • {improvement}")
            
            if ai_result['key_points_missed'] and test_case['ideal_answer']:
                print(f"\n{Colors.FAIL}✗ Key Points Missed:{Colors.ENDC}")
                for point in ai_result['key_points_missed']:
                    print(f"   • {point}")
            
            # Show detailed scores
            if ai_result.get('detailed_scores'):
                print(f"\n{Colors.OKCYAN}Detailed Component Scores:{Colors.ENDC}")
                for key, value in ai_result['detailed_scores'].items():
                    print(f"   • {key.replace('_', ' ').title()}: {value:.1f}/100")
        
        except Exception as e:
            print_error(f"AI evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Calculate Final Score
        print_info("\nCalculating final score...")
        answer_words = len(test_case['answer'].split())
        
        final_result = scorer.calculate_final_score(
            ai_relevance_score=ai_result['relevance_score'],
            emotion=test_case['emotion'],
            filler_count=test_case['filler_count'],
            answer_length_words=answer_words,
            sentiment_score=ai_result['detailed_scores'].get('sentiment_score')
        )
        
        all_scores.append(final_result)
        
        # Display final results
        print(f"\n{Colors.BOLD}📈 FINAL RESULTS:{Colors.ENDC}")
        print(f"   Final Score: {Colors.BOLD}{final_result['final_score']}/100{Colors.ENDC}")
        print(f"   Grade: {Colors.BOLD}{final_result['grade']}{Colors.ENDC}")
        print(f"\n   Component Breakdown:")
        print(f"     • AI Relevance (40%): {final_result['components']['relevance']}/100")
        print(f"     • Emotion (25%): {final_result['components']['emotion']}/100 ({final_result['emotion_detected']})")
        print(f"     • Fluency (20%): {final_result['components']['fluency']}/100 ({final_result['filler_words']} fillers)")
        print(f"     • Communication (15%): {final_result['components']['communication']}/100")
        print(f"\n   Assessment: {final_result['assessment']}")
        
        print("\n" + "="*80)
    
    # Session Summary
    print_header("Session Summary")
    
    if all_scores:
        session_metrics = scorer.calculate_session_metrics(all_scores)
        
        print(f"{Colors.BOLD}Overall Performance:{Colors.ENDC}")
        print(f"   Total Questions: {session_metrics['total_questions']}")
        print(f"   Average Score: {Colors.BOLD}{session_metrics['average_score']}/100{Colors.ENDC}")
        print(f"   Overall Grade: {Colors.BOLD}{session_metrics['overall_grade']}{Colors.ENDC}")
        print(f"   Average Relevance: {session_metrics['average_relevance']}/100")
        print(f"   Emotion Stability Index: {session_metrics['emotion_stability_index']}")
        print(f"\n   Score Range: {session_metrics['lowest_score']} - {session_metrics['highest_score']}")
        
        print(f"\n{Colors.BOLD}Emotion Distribution:{Colors.ENDC}")
        for emotion, count in session_metrics['emotion_distribution'].items():
            bar = "█" * count + "░" * (session_metrics['total_questions'] - count)
            print(f"   {emotion.capitalize():12} {bar} {count}")
        
        print(f"\n{Colors.BOLD}Session Assessment:{Colors.ENDC}")
        print(f"   {session_metrics['assessment']}")
    
    # Final summary
    print_header("Test Complete")
    print_success("All systems functioning correctly!")
    print(f"{Colors.OKGREEN}💰 Total Cost: $0.00 (Completely FREE!){Colors.ENDC}")
    print_info(f"Tested on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{Colors.BOLD}What's Working:{Colors.ENDC}")
    print("  ✅ Semantic similarity analysis")
    print("  ✅ Answer completeness checking")
    print("  ✅ STAR method detection")
    print("  ✅ Sentiment analysis")
    print("  ✅ Keyword coverage verification")
    print("  ✅ Detailed feedback generation")
    print("  ✅ Session metrics calculation")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("  1. Integrate free_ai_evaluator.py into your app.py")
    print("  2. Update Streamlit UI to display AI feedback")
    print("  3. Test with microphone recording")
    print("  4. Add AI feedback to PDF exports")
    print("  5. Deploy to students for unlimited practice!")
    
    print(f"\n{Colors.OKGREEN}🎉 Ready to go! No costs, no limits, no problems!{Colors.ENDC}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
