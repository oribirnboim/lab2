import numpy as np
from numpy import sin, cos, abs, angle, sum, array, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt


class Wave:
    def __init__(self, waves: list['Wave'] =[]):
        self.waves = waves
    
    def add_wave(self, wave: 'Wave'):
        self.waves.append(wave)

    def amplitude(self, r: array) -> float:
        return abs(self.get_complex(r))
    
    def intensity(self, r: array) -> float:
        return(self.amplitude(r)**2)

    def phase(self, r: array) -> complex:
        return angle(self.get_complex(r))
    
    def get_complex(self, r: array) -> complex:
        if len(self.waves) == 0: raise AttributeError
        return sum([w.get_complex(r) for w in self.waves])
    
    def plot_plane_intensity(self, center, side_length, num_points, theta):
        points = generate_3d_grid(center, side_length, num_points, theta)
        intensity = np.zeros((points.shape[0], points.shape[1]))
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                r = points[i, j]
                intensity[i, j] = self.intensity(r)
        plt.figure()
        plt.imshow(intensity, extent=(-side_length/2, side_length/2, -side_length/2, side_length/2), origin='lower', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Intensity on the plane')
        plt.show()

def generate_3d_grid(center, side_length, num_points, theta):
    # Generate a grid of points in the xy-plane
    linspace = np.linspace(-side_length/2, side_length/2, num_points)
    x, y = np.meshgrid(linspace, linspace)
    
    # Convert spherical coordinates (theta, phi) to Cartesian coordinates for the normal vector
    normal = array([cos(theta), sin(theta), 0])
    
    # Find two orthogonal vectors in the plane
    v2 = array([0, 0, 1])
    v1 = np.cross(normal, v2)
    
    # Generate the 3D points
    points = np.zeros((num_points, num_points, 3))
    for i in range(num_points):
        for j in range(num_points):
            points[i, j] = center + x[i, j] * v1 + y[i, j] * v2
            
    return points


class PointWave(Wave):
    def __init__(self, power: float, r0: array, source_phase: float, wavelength: float):
        super().__init__()
        self.power = power
        self.r0 = r0
        self.source_phase = source_phase
        self.wavelength = wavelength

    def amplitude(self, r: array) -> float:
        radius = np.linalg.norm(r-self.r0)
        return self.power / radius

    def phase(self, r: array) -> float:
        radius = norm(r-self.r0)
        return self.source_phase + 2 * pi * radius / self.wavelength
    
    def get_complex(self, r: array) -> complex:
        a = self.amplitude(r)
        p = self.phase(r)
        return a*complex(cos(p), sin(p))


class PlaneWave(Wave):
    def __init__(self, wavelength: float, amplitude: float, source_phase: float, r0: array, normal: array):
        super().__init__()
        self.amp = amplitude
        self.wavelength = wavelength
        self.normal = normal
        self.r0 = r0
        self.source_phase = source_phase

    def amplitude(self, r: array) -> float:
        return self.amp

    def phase(self, r: array) -> float:
        return self.source_phase + 2 * pi * np.dot(r-self.r0, self.normal)/self.wavelength
    
    def get_complex(self, r: array) -> complex:
        a = self.amplitude(r)
        p = self.phase(r)
        return a*complex(cos(p), sin(p))
    


# Example usage
if __name__ == "__main__":
    wavelength = 1
    # p = PointWave(power=1, r0=array([10, -1, 0]), source_phase=0, wavelength=wavelength)
    # p = Wave([p])
    # c = PointWave(power=1, r0=array([10, 1, 0]), source_phase=0, wavelength=wavelength)
    # p.add_wave(c)
    p = Wave([PlaneWave(wavelength=wavelength, amplitude=1, source_phase=0, r0=array([0,0,0]), normal=array([1, 1, 0])/np.sqrt(2))])
    p.add_wave(Wave([PlaneWave(wavelength=wavelength, amplitude=1, source_phase=0, r0=array([0,0,0]), normal=array([1, -1, 0])/np.sqrt(2))]))
    center = array([0, 0, 0])
    side_length = 2
    num_points = 300
    theta = 0
    p.plot_plane_intensity(center=center, side_length=side_length, num_points=num_points, theta=theta)

