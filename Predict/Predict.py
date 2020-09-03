# Prediction Compiled Process Model
from Temp_Prediction import Temp_Prediction
from Wind_Prediction import Wind_Prediction
from Clarity_Prediction import Clarity_Prediction
from Load_Prediction import Load_Prediction

# Identify and Stnadrize weather variables
T = Temp_Prediction + 273.15
W = Wind_Prediction
C = Clarity_Prediction

# Silicon cells eq of production
Si = C*(2 - ((4.37*(10**-4)*(T**2))/(T+636)))
print("silicon cell = " , Si)

# Approximated Production per meter squared
m = Si*100
print("solar production per meter squared = " , m)

# Wind Turbine eq of production
p = 1.225
Diameter = 150
R = Diameter/2
Pi = 3.1416
A = Pi*(R**2)
Cp = 0.593
WT = 0.5*p*A*Cp*(W**3)/1000

# Sizing the PV farm & calculating its Production
Farm_Size = 1000*1000
PV_Production = Farm_Size*m/1000
print("PV Farm Production in KWH = " , PV_Production)

# Determining number of Wind Turbines & calculating their Production
Wind_Turbines = 100
WT_Prpduction = WT*Wind_Turbines
print("Single Wind Turbine Production in KWH = ", WT)
print("Wind Turbine Production in KWH = " , WT_Prpduction)

# Calculating Total Production and identifying givin Demand
Total_Production = (PV_Production + WT_Prpduction)/1000
Load_Demand = Load_Prediction
print("Demand Production in MWH = ", Load_Demand)
print("Total Production in MWH = " , Total_Production)

# Calculate Surplus power or shortage of power depending on the difference
if Total_Production > Load_Demand :
    Surplus_Power = Total_Production - Load_Demand
    print("Surplus power in MWH = ", Surplus_Power)

else :
    if Total_Production < Load_Demand :
        Shortage_Power = Load_Demand - Total_Production
        print("Shortage of power in MWH = ", Shortage_Power)

    else:
        print("Production = Demand")

