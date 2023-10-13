from machine_learning.dataset.linreg import Reta
from machine_learning.models.multilayer_perceptron import MLP

ds = Reta(10, -2, 1000)
model = MLP(input_layer=1, hidden_layer=(0, ))

model.fit(dataset_training=ds, num_epochs=100, num_batches=16)

print(f'Valores predefinidor {10}, {-2}')