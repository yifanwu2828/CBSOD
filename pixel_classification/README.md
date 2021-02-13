# One-vs-all Logistic Regression Approach

Three binary logistic Regression model were trained based on RGB color space to classify (Red=1, Green=2, Blue=3)

A blue-bin detection problem consists of three problems:
pixel classification, image segmentation and Region of In-
terests classification. Three binary logistic regression models
are combined into an one-vs-all logistic regression model. In
each binary logistic regression, given a sample pixel x ∈ R 3 ,
a logistic sigmoid function maps the continuous value ω T x
to a Bernoulli probability mass function for a binary label
y ∈ {−1, 1}. Three parameters (ω red
, ω green
, ω blue
) were
optimized individually with respect to 3 target colors (red,
green, blue).


![plot](/Results/blue.png)
![plot](/Results/green.png)
![plot](/Results/red.png)
