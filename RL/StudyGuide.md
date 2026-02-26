# Step by step
1. Pre-Training
2. Post-Training (Alignment + Specialization)
 - Supervised Fine-Tuning (SFT)
 - Preference Training (Reward Model)
 - RLHF (Reinforcement Learning from Human Feedback)

# LORA
LoRA is a parameter-efficient training technique.
You can apply LoRA during:
 - SFT
 - Reward model training
 - RLHF
 - Even domain adaptation
```python
SFT (full fine-tune OR LoRA)
Reward model training (full OR LoRA)
RLHF (full OR LoRA)
```