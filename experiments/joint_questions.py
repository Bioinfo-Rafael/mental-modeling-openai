"""One question mapping shared by Exp.03_1, 05, and 06.

Prompt text remains owned by upstream/LLM-Xavier/llm_x/feedback.py.
"""
JOINT_QUESTIONS = {
    "MountainCar-v0": {
        "next-action": "next_action_prediction",
        "last-action": "last_action_prediction",
        "next-state": "next_state_prediction_more_options_joint",
        "last-state": "last_state_prediction_more_options_joint",
    },
    "Pendulum-v1": {
        "next-action": "next_action_prediction_continuous_joint",
        "last-action": "last_action_prediction_continuous_joint",
        "next-state": "next_state_prediction_more_options_joint",
        "last-state": "last_state_prediction_more_options_joint",
    },
}
