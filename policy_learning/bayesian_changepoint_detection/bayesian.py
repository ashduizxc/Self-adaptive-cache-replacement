from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
from bayesian_changepoint_detection.generate_data import generate_normal_time_series
import bayesian_changepoint_detection.online_likelihoods as online_ll
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

partition,data = generate_normal_time_series(7,50,200)

hazard = partial(constant_hazard, 250)
R, maxex = online_changepoint_detection(data, hazard, online_ll.StudentT(alpha=0.1, beta=.01, kappa=1,mu=0))
fig, ax = plt.subplots(2, figsize=[18,16], sharex=True)
ax[0].plot(data)
Nw = 10

print(R[Nw,Nw:-1])
ax[1].plot(R[Nw,Nw:-1])
plt.show()

