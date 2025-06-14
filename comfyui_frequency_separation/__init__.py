from .nodes import FrequencySplit, FrequencyRecombine

NODE_CLASS_MAPPINGS = {
    "FrequencySplit": FrequencySplit,
    "FrequencyRecombine": FrequencyRecombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrequencySplit": "Frequency Split",
    "FrequencyRecombine": "Frequency Recombine",
}
