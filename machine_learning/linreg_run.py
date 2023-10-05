from machine_learning.dataset.linreg import Reta
from machine_learning.models.linear_regress import LinearRegress

ds = Reta(10, -2, 1000)
model = LinearRegress()

model.fit(ds, 10, 16)

print(f'Valores predefinidor {10}, {-2}')