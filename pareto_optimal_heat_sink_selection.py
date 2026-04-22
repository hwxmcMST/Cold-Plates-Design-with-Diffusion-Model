import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

data = pd.read_csv("generated_predictions_denormalized.csv")
Rth = data["Thermal Resistance (K/W)"].values
DeltaP = data["Pressure Drop (Pa)"].values
Nu = data["Nusselt Number"].values
geom_id = data["filename"]
print(f"Loaded {len(Rth)} geometry samples.")

def normalize_minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

Rth_star = normalize_minmax(Rth)
DeltaP_star = normalize_minmax(DeltaP)
Nu_star = (np.max(Nu) - Nu) / (np.max(Nu) - np.min(Nu))

F = np.column_stack([Rth_star, DeltaP_star, Nu_star])

nds = NonDominatedSorting()

pareto_idx = nds.do(F, only_non_dominated_front=True)
print(f"Found {len(pareto_idx)} Pareto-optimal geometries.")
pareto_data = data.iloc[pareto_idx]
pareto_data.to_csv("pareto_optimal_geometries.csv", index=False)


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Rth, DeltaP, Nu, color='gray', alpha=0.4, label='All Geometries')
ax.scatter(Rth[pareto_idx], DeltaP[pareto_idx], Nu[pareto_idx],
           color='red', s=60, label='Pareto Front')
ax.set_xlabel('Thermal Resistance R_th (K/W)')
ax.set_ylabel('Pressure Drop Δp (Pa)')
ax.set_zlabel('Nusselt Number Nu')
ax.set_title('Pareto Front for Thermal–Hydraulic Performance')
ax.legend()
plt.tight_layout()
plt.show()

# ===============================================================
# STEP 6 — Optional 2D projections for paper visualization
# ===============================================================
plt.figure(figsize=(6,5))
plt.scatter(Rth, DeltaP, c=Nu, cmap='viridis', alpha=0.6)
plt.colorbar(label='Nusselt Number Nu')
plt.scatter(Rth[pareto_idx], DeltaP[pareto_idx], color='red', label='Pareto Front')
plt.xlabel("Thermal Resistance R_th (K/W)")
plt.ylabel("Pressure Drop Δp (Pa)")
plt.title("Pareto Front Projection (Colored by Nu)")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================================================
# Additional 2D Projection: R_th vs Nu (color = Δp)
# ===============================================================
plt.figure(figsize=(6,5))
plt.scatter(Rth, Nu, c=DeltaP, cmap='plasma', alpha=0.6)
plt.colorbar(label='Pressure Drop Δp (Pa)')
plt.scatter(Rth[pareto_idx], Nu[pareto_idx], color='red', label='Pareto Front')
plt.xlabel("Thermal Resistance R_th (K/W)")
plt.ylabel("Nusselt Number Nu")
plt.title("Pareto Front Projection (Colored by Δp)")
plt.legend()
plt.tight_layout()
plt.show()

