import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from typing import *


sim_name = 'ez'
filename = 'meep_sim'

WL = 0.6328 #unit length is micron
f = 10
rings = 5
absorbing_material = mp.Medium(epsilon=100)
cell = mp.Vector3(1.5*f,20,0)


def create_2d_mask() -> list:
    def r(n): return np.sqrt(n*WL*f + 1/4*n**2*WL**2)
    res = []
    res = [mp.Block(center=mp.Vector3(-4/12*cell[0]), size=mp.Vector3(cell[0]/24, mp.inf, mp.inf), material=absorbing_material)]
    for i in range(rings):
        r_low = r(2*i)
        r_high = r(2*i+1)
        res.append(mp.Block(center=mp.Vector3(-4/12*cell[0], (r_high+r_low)/2), size=mp.Vector3(cell[0]/24, r_high-r_low, mp.inf), material=mp.vacuum))
        res.append(mp.Block(center=mp.Vector3(-4/12*cell[0], -(r_high+r_low)/2), size=mp.Vector3(cell[0]/24, r_high-r_low, mp.inf), material=mp.vacuum))
    return res
    return []

geometry = create_2d_mask()

def create_2d_sources() -> list:
    res = []
    plane_source = mp.Source(mp.ContinuousSource(wavelength=WL, width=20),
                             component=mp.Ez,
                             center=mp.Vector3(-5/12*cell[0]),
                             size=mp.Vector3(0, cell[1]))
    res.append(plane_source)
    return res


sources = create_2d_sources()


pml_layers = [mp.PML(1.0)]
resolution = 10


def main():
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    sim.run(mp.at_beginning(mp.output_epsilon), 
            mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
            until=200)
    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric).transpose()
    plt.imshow(eps_data, cmap='binary')
    plt.show()

    e_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez).transpose()
    plt.imshow(e_data, interpolation='spline36', cmap='RdBu', alpha=0.9)
    plt.show()
    

    
    # command = f'h5ls {filename}-{sim_name}.h5'
    # subprocess.run(command, shell=True, capture_output=False)
    # command = f'h5topng -t 0:332 -R -Zc dkbluered -a yarg -A {filename}-eps-000000.00.h5 {filename}-{sim_name}.h5'
    # subprocess.run(command, shell=True, capture_output=False)



if __name__ == "__main__":
    main()