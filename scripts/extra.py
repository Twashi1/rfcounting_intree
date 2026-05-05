import math

avg_temp_pe_errors = [
    1.75,
    3.63,
    5.56,
    0.09,
    0.77,
    0.1,
    5.1,
    0.77,
    0.09,
    1.44,
    5.39,
    5.69,
    0.99,
    0.59,
]

rmspe = math.sqrt(
    sum(err * err for err in avg_temp_pe_errors) / len(avg_temp_pe_errors)
)

print(f"RMSPE in avg temperature: {rmspe}")
