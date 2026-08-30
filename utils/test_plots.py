import numpy as np

from utils.plots import TrainingPlots

plot = TrainingPlots()

plot.accuracy_plot(

    [70,75,80,88,92],

    [68,73,79,86,91]

)

plot.loss_plot(

    [1.1,0.9,0.6,0.4,0.2],

    [1.2,1.0,0.7,0.5,0.3]

)

y_true = np.random.randint(

    0,

    4,

    100

)

y_score = np.random.rand(

    100,

    4

)

plot.roc_curve_plot(

    y_true,

    y_score

)

plot.precision_recall_plot(

    y_true,

    y_score

)
