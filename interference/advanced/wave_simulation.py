import numpy as np
from numpy import sin, cos, abs, angle, sum, array, pi, power
from numpy.linalg import norm
import matplotlib.pyplot as plt
from typing import Callable


WL = 632.8*power(10., -9) #real wavelength




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
    
    def get_plane_intensity(self, points) -> array:
        intensity = np.zeros((points.shape[0], points.shape[1]))
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                r = points[i, j]
                intensity[i, j] = self.intensity(r)
        return intensity

    def plot_plane_intensity(self, intensity: array) -> None:
        plt.figure()
        plt.imshow(intensity, extent=(-side_length/2, side_length/2, -side_length/2, side_length/2), origin='lower', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Intensity')
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
    def __init__(self, amp: float, r0: array, source_phase: float, wavelength: float):
        super().__init__()
        self.amp = amp
        self.r0 = r0
        self.source_phase = source_phase
        self.wavelength = wavelength

    def amplitude(self, r: array) -> float:
        radius = np.linalg.norm(r-self.r0)
        return self.amp / radius

    def phase(self, r: array) -> float:
        radius = norm(r-self.r0)
        return self.source_phase + 2 * pi * radius / self.wavelength
    
    def get_complex(self, r: array) -> complex:
        a = self.amplitude(r)
        p = self.phase(r)
        return a*complex(cos(p), sin(p))


class PlaneWave(Wave):
    def __init__(self, amp: float, r0: array, source_phase: float, wavelength: float, normal: array):
        super().__init__()
        self.amp = amp
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


class CustomWave(Wave):
    def __init__(self, f: Callable[[float, float], complex], waves: list[Wave] = []):
        super().__init__(waves)
    


def apply_mask(points: array, intensity: array, wave: 'Wave') -> Wave:
    res = Wave()
    x, y = np.shape(intensity)
    for i in range(x):
        for j in range(y):
            r = points[i][j]
            mask = intensity[i][j]
            amp = wave.amplitude(r)
            phase = wave.phase(r)
            p = PointWave(amp=amp*mask, r0=r, source_phase=phase, wavelength=WL)
            res.add_wave(p)
    return res


# Example usage
if __name__ == "__main__":
    num_points = 40
    d = 2*WL
    points = [array([x, y, 0]) for x, y in [(0,0)]]
    image = Wave([PointWave(amp=1*WL, r0=point, source_phase=0, wavelength=WL) for point in points])
    plane1 = PlaneWave(amp=1, r0=array([0,0,0]), source_phase=0, wavelength=WL, normal=array([0.8,-0.6,0])/np.sqrt(2))
    plane2 = PlaneWave(amp=1, r0=array([0,0,0]), source_phase=0, wavelength=WL, normal=array([0.8,0.6,0])/np.sqrt(2))
    plane3 = PlaneWave(amp=1, r0=array([0,0,0]), source_phase=0, wavelength=WL, normal=array([0.8,0,-0.6])/np.sqrt(2))
    plane4 = PlaneWave(amp=1, r0=array([0,0,0]), source_phase=0, wavelength=WL, normal=array([0.8,0,0.6])/np.sqrt(2))
    reference_wave = Wave([plane1, plane2, plane3, plane4])
    initial_wave = Wave([image, reference_wave])

    center = array([d, 0, 0])
    side_length = 5*WL
    theta = 0
    mask_points = generate_3d_grid(center, side_length, num_points, theta)

    intensity = initial_wave.get_plane_intensity(mask_points)
    initial_wave.plot_plane_intensity(intensity)

    masked_wave = apply_mask(mask_points, intensity, reference_wave)

    center = array([2*d, 0, 0])
    side_length = 2*power(10., -6)
    theta = 0
    image_points = generate_3d_grid(center, side_length, num_points, theta)

    final_intensity = masked_wave.get_plane_intensity(image_points)
    masked_wave.plot_plane_intensity(final_intensity)