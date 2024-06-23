import numpy as np
from models.model import Model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ProjectMLPRegressor(Model):
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, activation = 'relu', hidden_layers = (100, ), combine_train_and_val = True):

        
        # call to super class constructor
        super().__init__(X_train, X_test, X_val, y_train, y_test, y_val, combine_train_and_val)


        self.activation = activation
        self.hidden_layers = hidden_layers
    


    def train(self):
        print(f"\n> Launch : MLP Regessor -> hidden-layer = {self.hidden_layers}\n")
        # Create the MLPRegressor model
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layers, activation=self.activation, solver='adam', max_iter=1000, random_state=42)

        # Train the model
        self.model.fit(self.X_train, self.y_train)
