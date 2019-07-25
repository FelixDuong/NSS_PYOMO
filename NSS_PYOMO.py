from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Y = [0.035, 0.0358, 0.0374, 0.0405, 0.0434, 0.0488, 0.0536, 0.0598, 0.0675, 0.0723]
# Creation of a Concrete Model
model = ConcreteModel(name="NSS Problem")
## Define sets ##
#  Sets
#       i   canning plants   / seattle, san-diego /
#       j   markets          / new-york, chicago, topeka / ;
model.Yields = Set(initialize=['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y'])
model.T = Param(model.Yields, initialize={'3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '15Y': 15, '20Y': 20})
model.Y = Param(model.Yields, initialize={'3M': 0.035, '6M': 0.0358, '1Y': 0.0374, '2Y': 0.0405, '3Y': 0.0434, '5Y': 0.0488, '7Y': 0.0536, '10Y': 0.0598, '15Y': 0.0675, '20Y': 0.0723}, mutable=True)
j = 0
df = pd.read_csv("daily_yield.txt", sep=" ", header=None)

df = np.asarray(df)
for i in model.Yields:
    model.Y[i] = float(df[j].item())
    j += 1

model.Beta1 = Var(initialize=model.Y['3M'].value, bounds=(model.Y['3M'].value, 1000))
model.Beta2 = Var(initialize=0.1)
model.Beta3 = Var(initialize=-0.1)
model.Beta4 = Var(initialize=-0.1)
model.Lambda1 = Var(initialize=0.1, bounds=(0.000001, 1000))
model.Lambda2 = Var(initialize=0.1, bounds=(0.000001, 1000))


# Must keep orginal form of vars to solve the problem , PYOMO only accepts vars in orginal form
def nss(model, T):
    return model.Beta1 + model.Beta2 * ((1 - exp(-T) ** (1/model.Lambda1))/ (T / model.Lambda1)) + model.Beta3 * \
           (((1 - exp(-T) ** (1/model.Lambda1)) / (T / model.Lambda1)) - (exp(-T) ** (1/model.Lambda1))) + model.Beta4 * (((1 - exp(-T) ** (1/model.Lambda2)) / (T / model.Lambda2)) - (exp(-T) ** (1/model.Lambda2)))


def constraint_rule(model):
    return model.Beta1 + model.Beta2 >= 1e-6


def contraint_nss(model):
    return nss(model, model.Y['20Y'].value) >= 1e-10
def contraint_obj(model):
    return sum((nss(model, model.T[i]) - model.Y[i]) ** 2 for i in model.Yields) <= 1e-9

def objective_rule(model):
    return sum((nss(model, model.T[i]) - model.Y[i]) ** 2 for i in model.Yields)

model.nss = Constraint(rule=contraint_nss)
model.diff = Constraint(rule=constraint_rule)
model.obj_con = Constraint(rule=contraint_obj)
model.objective = Objective(rule=objective_rule, sense=minimize, doc='NSS Yield Curve')


# Display of the output ##
# Display x.l, x.m ;
def pyomo_postprocess(options=None, instance=None, results=None):
    print("Beta1 = ", model.Beta1.value)
    print("Beta2 = ", model.Beta2.value)
    print("Beta3 = ", model.Beta3.value)
    print("Beta4 = ", model.Beta4.value)
    print("Lambda1 = ", model.Lambda1.value)
    print("Lambda2 = ", model.Lambda2.value)


# solver = SolverFactory('cplex')
# solver.options['max_iter'] = 10000
# results = solver.solve(model, tee=True)
opt = SolverFactory("conopt")
# opt = SolverFactory("/home/felix/Documents/amplide.linux64/cplex")
# opt.options['max_iter'] = 10000
results = opt.solve(model, tee=True, keepfiles=True, logfile="process.log", timelimit=120)
# sends results to stdout

results.write()
print("\nDisplaying Solution\n" + '-' * 60)
pyomo_postprocess(None, model, results)
import numpy as np

x0 = np.linspace(0.1, 30, 300)
y0 = []
for i in range(len(x0)):
    y0.append(value(nss(model, float(x0[i]))))
plt.plot(y0)
plt.show()
