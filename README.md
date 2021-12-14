## Scania Truck Failures
### Predict whether a failure is related to the air pressure system

#### Problem Statement:

    The Air Pressure System (APS) is a critical component of a heavy-duty vehicle that uses
    compressed air to force a piston to provide pressure to the brake pads, slowing the vehicle
    down. The benefits of using an APS instead of a hydraulic system are the easy availability
    and long-term sustainability of natural air.This is a Binary Classification problem, in
    which the affirmative class indicates that the failure was caused by a certain component of
    the APS, while the negative class indicates that the failure was caused by something else.

#### Data Description:

    The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurised air that are utilized in various functions in a truck, such as braking and gear changes. The datasets' positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS. The data consists of a subset of all available data, selected by experts.

    Link: https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks

    Cost-metric of miss-classification:

    The total cost of a prediction model the sum of 'Cost_1' multiplied by the number of Instances with type 1 failure and 'Cost_2' with the number of instances with type 2 failure,resulting in a 'Total_cost'.
    In this case Cost_1 refers to the cost that an unnessecary check needs to be done by an mechanic at an workshop, while Cost_2 refer to the cost of missing a faulty truck, which may cause a breakdown.

    Total_cost = Cost_1*No_Instances + Cost_2*No_Instances.

    Number of Instances:
    The training set contains 60000 examples in total in which 59000 belong to the negative class and 1000 positive class.The test set contains 16000 examples.

    Number of Attributes: 171
