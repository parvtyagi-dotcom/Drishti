"""AI analyst layer: agents that read Drishti's events and brief incidents.

Detection, alarms and the siren stay rule-based and instant in the camera
pipeline. Everything in this package sits after that: it reads events,
explains them, and proposes actions that only take effect when an operator
approves them. Each step the agents take is written to the same
tamper-evident event log as the detections themselves.
"""
