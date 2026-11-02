"""Fixed episode and windows; model IDs retain the prior experiments' defaults."""
EPISODE = 'mhi_v1.3_ic100601_win'
CENTERS = (45, 85, 145, 245, 310)
MODELS = {'sol': 'gpt-5.6-sol', 'terra': 'gpt-5.6-terra', 'luna': 'gpt-5.6-luna'}
ACTION_FIELDS = ('altitude_action', 'heading_action', 'speed_action')
REFERENCE_URL = 'https://docs.google.com/spreadsheets/d/1EIgeVnxUKSuLsN2aV1Vqb0Tmjwi7bY85CtlCrYRPyIE/edit?usp=sharing'
