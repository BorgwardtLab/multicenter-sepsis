"""Methods for implementing the physionet 2019 sepsis recognition score.

The python methods to compute the physionet score are based on the code at
https://github.com/physionetchallenges/evaluation-2019

Which was published under the following licence:

BSD 2-Clause License

Copyright (c) 2019, PhysioNet
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from .sklearn_utils import nanany

def compute_prediction_utility(labels, predictions, dt_early=-12,
                               dt_optimal=-6, dt_late=3.0, max_u_tp=1,
                               min_u_fn=-2, u_fp=-0.05, u_tn=0,
                               check_errors=True):
    """Compute utility score of physionet 2019 challenge."""
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1) and not np.isnan(label):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1) and not np.isnan(prediction):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if nanany(labels):
        is_septic = True
        # Change this as we do not have shifted labels in our setup
        # t_sepsis = np.argmax(labels) - dt_optimal
        t_sepsis = np.nanargmax(labels)
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and (predictions[t] == 1.):
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and (predictions[t] == 1.):
                u[t] = u_fp
            # FN
            elif is_septic and (predictions[t] == 0.):
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and (predictions[t] == 0.):
                u[t] = u_tn
            else:
                # predictions[t] is probably np.NaN
                pass

    # Find total utility for patient.
    return np.sum(u)


def physionet2019_utility(y_true, y_score):
    """Compute physionet 2019 Sepsis eary detection utility.

    Args:
        y_true:
        y_score:

    Returns:
        Normalized physionet score

    """
    dt_early = -12
    # dt_optimal = -6
    dt_late = 3.0

    utilities = []
    best_utilities = []
    inaction_utilities = []

    for labels, observed_predictions in zip(y_true, y_score):
        observed_predictions = np.round(observed_predictions)
        num_rows = len(labels)
        best_predictions = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if nanany(labels):
            # Change this as we do not have shifted labels in our setup
            # t_sepsis = np.argmax(labels) - dt_optimal
            t_sepsis = np.nanargmax(labels)
            pred_begin = int(max(0, t_sepsis + dt_early))
            pred_end = int(min(t_sepsis + dt_late + 1, num_rows))
            best_predictions[pred_begin:pred_end] = 1
            best_predictions[np.isnan(labels)] = np.NaN

        utilities.append(
            compute_prediction_utility(labels, observed_predictions))
        best_utilities.append(
            compute_prediction_utility(labels, best_predictions))
        inaction_utilities.append(
            compute_prediction_utility(labels, inaction_predictions))

    unnormalized_observed_utility = sum(utilities)
    unnormalized_best_utility = sum(best_utilities)
    unnormalized_inaction_utility = sum(inaction_utilities)
    normalized_observed_utility = (
        (unnormalized_observed_utility - unnormalized_inaction_utility)
        / (unnormalized_best_utility - unnormalized_inaction_utility)
    )
    return normalized_observed_utility
