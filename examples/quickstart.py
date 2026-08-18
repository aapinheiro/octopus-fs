"""Smallest end-to-end example. Keep it runnable — CI should execute this file.

Once the arms exist, this doubles as the smoke test for the whole library.
"""

# TODO(you): fill in after v0.1 arms are implemented.
#
# from sklearn.datasets import make_classification
# import pandas as pd
# from octopus import Octopus
#
# X, y = make_classification(n_samples=2000, n_features=30, n_informative=6,
#                            n_redundant=4, random_state=0)
# X = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(30)])
# y = pd.Series(y, name="target")
#
# result = Octopus(arms="fast", random_state=42).fit(X, y)
# print(result.consensus.top(10))
# result.to_html("reports/quickstart.html")
