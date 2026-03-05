#!/usr/bin/env python3
"""
Example: How to implement real LLM judge API calls
==================================================
This shows how to replace the heuristic fallback with actual API calls to GPT-4 or Claude.

SETUP REQUIRED:
1. Install: pip install openai anthropic
2. Set environment variables:
   - export OPENAI_API_KEY="your-openai-key"
   - export ANTHROPIC_API_KEY="your-anthropic-key"
"""

import os
import logging
from typing import Optional

# Optional imports - only if you want to use real LLM judge
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not installed. Run: pip install openai")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Anthropic not installed. Run: pip install anthropic")

logger = logging.getLogger(__name__)

class RealLLMJudge:
    """Implementation of actual LLM judge API calls"""
    
    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("OpenAI library not installed")
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4"
            
        elif self.provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("Anthropic library not installed")
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-3-sonnet-20240229"
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def judge_correctness(self, task: str, response: str, ground_truth: str = None) -> float:
        """
        Use real LLM to judge response correctness
        
        Returns score from 0.0 to 1.0
        """
        
        judge_prompt = f"""You are an expert evaluator. Assess the correctness of this response.

Task: {task}

Response to evaluate: {response}

Ground truth/Expected answer: {ground_truth or "Use your expertise to determine correctness"}

Rate the correctness from 0.0 to 1.0 where:
- 1.0 = Completely correct, accurate, and addresses the task fully
- 0.8 = Mostly correct with minor errors or omissions  
- 0.6 = Partially correct but missing key elements
- 0.4 = Some correct elements but significant issues
- 0.2 = Mostly incorrect with few correct elements
- 0.0 = Completely incorrect or irrelevant

Provide only the numerical score (0.0-1.0):"""

        try:
            if self.provider == "openai":
                return self._call_openai(judge_prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(judge_prompt)
        except Exception as e:
            logger.error(f"LLM judge API call failed: {e}")
            return 0.5  # Fallback to uncertain
    
    def _call_openai(self, prompt: str) -> float:
        """Call OpenAI GPT-4 for judgment"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Respond only with a number."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1  # Low temperature for consistent scoring
        )
        
        score_text = response.choices[0].message.content.strip()
        
        # Extract numerical score
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to valid range
        except ValueError:
            logger.warning(f"Could not parse LLM judge score: {score_text}")
            return 0.5
    
    def _call_anthropic(self, prompt: str) -> float:
        """Call Anthropic Claude for judgment"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        score_text = response.content[0].text.strip()
        
        # Extract numerical score
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to valid range
        except ValueError:
            logger.warning(f"Could not parse LLM judge score: {score_text}")
            return 0.5


# Enhanced AdvancedEvaluator that uses real LLM judge
class EnhancedAdvancedEvaluator:
    """
    AdvancedEvaluator with real LLM judge implementation
    
    This shows how to modify the original AdvancedEvaluator to use actual API calls
    """
    
    def __init__(self, use_llm_judge: bool = True, judge_provider: str = "openai", judge_model: str = None):
        self.use_llm_judge = use_llm_judge
        
        if self.use_llm_judge:
            try:
                self.llm_judge = RealLLMJudge(provider=judge_provider, model=judge_model)
                logger.info(f"Initialized {judge_provider} LLM judge")
            except Exception as e:
                logger.error(f"Failed to initialize LLM judge: {e}")
                self.use_llm_judge = False
                logger.warning("Falling back to heuristic evaluation")
    
    def _llm_judge_correctness(self, task: str, response: str, ground_truth: str) -> float:
        """
        Enhanced LLM judge with real API calls
        
        This replaces the heuristic fallback in the original implementation
        """
        
        if not self.use_llm_judge:
            return self._heuristic_correctness_judge(response, ground_truth)
        
        try:
            return self.llm_judge.judge_correctness(task, response, ground_truth)
        except Exception as e:
            logger.error(f"LLM judge failed, falling back to heuristic: {e}")
            return self._heuristic_correctness_judge(response, ground_truth)
    
    def _heuristic_correctness_judge(self, response: str, ground_truth: str) -> float:
        """Fallback heuristic when LLM judge is unavailable"""
        # Same implementation as in advanced_evaluation_system.py
        if not ground_truth:
            if len(response.strip()) < 20:
                return 0.2
            elif "error" in response.lower() or "fail" in response.lower():
                return 0.3
            elif len(response.split('\n')) >= 3:
                return 0.7
            else:
                return 0.5
        
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if len(truth_words) == 0:
            return 0.5
            
        overlap = len(response_words.intersection(truth_words))
        similarity = overlap / len(truth_words)
        
        return min(similarity, 1.0)


# Example usage
def example_usage():
    """Example showing how to use real LLM judge"""
    
    print("=== LLM Judge Implementation Example ===\n")
    
    # Check if API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("⚠️  No API keys found!")
        print("Set environment variables:")
        print("export OPENAI_API_KEY='your-key'")
        print("export ANTHROPIC_API_KEY='your-key'")
        print("\nUsing heuristic fallback for this demo...")
        use_real_judge = False
        provider = "heuristic"
    else:
        use_real_judge = True
        provider = "openai" if openai_key else "anthropic"
        print(f"✅ Found {provider.upper()} API key, using real LLM judge")
    
    # Initialize evaluator
    try:
        evaluator = EnhancedAdvancedEvaluator(
            use_llm_judge=use_real_judge,
            judge_provider=provider
        )
        
        # Test evaluation
        task = "Calculate 15% of 240, then add 18. Show your work."
        response = """To calculate 15% of 240, I'll work step by step:

1. Convert 15% to decimal: 15% = 0.15
2. Multiply: 240 × 0.15 = 36  
3. Add 18: 36 + 18 = 54

Therefore, the final answer is 54."""
        
        ground_truth = "The answer is 54, calculated by finding 15% of 240 (which is 36) and adding 18."
        
        print(f"\n📝 Task: {task}")
        print(f"🤖 Response: {response[:100]}...")
        print(f"✅ Ground Truth: {ground_truth}")
        
        score = evaluator._llm_judge_correctness(task, response, ground_truth)
        
        print(f"\n📊 LLM Judge Score: {score:.2f}")
        print(f"🎯 Method: {'Real LLM API' if use_real_judge else 'Heuristic Fallback'}")
        
        if score >= 0.8:
            print("✅ Excellent response quality")
        elif score >= 0.6:
            print("👍 Good response quality")
        else:
            print("⚠️  Needs improvement")
            
    except Exception as e:
        print(f"❌ Error in example: {e}")


if __name__ == "__main__":
    example_usage()