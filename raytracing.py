import numpy as np
from PIL import Image

# Vector class
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Vector addition
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    # Vector subtraction
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    # Scalar multiplication
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    # Dot product
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    # Normalize the vector
    def normalize(self):
        norm = np.sqrt(self.dot(self))
        return Vector(self.x / norm, self.y / norm, self.z / norm)

    # Convert to numpy array
    def to_np(self):
        return np.array([self.x, self.y, self.z])

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin      # Vector
        self.direction = direction.normalize()  # Vector

# Sphere class
class Sphere:
    def __init__(self, center, radius, color):
        self.center = center  # Vector
        self.radius = radius
        self.color = color    # Tuple (R, G, B)

    # Ray-sphere intersection
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return None  # No intersection
        else:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            if t1 > 0:
                return t1
            if t2 > 0:
                return t2
            return None

# Scene setup
spheres = [
    Sphere(Vector(0, -1, 3), 1, (255, 0, 0)),    # Red sphere
    Sphere(Vector(2, 0, 4), 1, (0, 0, 255)),     # Blue sphere
    Sphere(Vector(-2, 0, 4), 1, (0, 255, 0)),    # Green sphere
]

# Light source
light = Vector(5, 5, -10)

# Image dimensions
WIDTH = 400
HEIGHT = 300

# Field of view
FOV = np.pi / 3  # 60 degrees

def cast_ray(ray, objects, light):
    nearest_t = float('inf')
    nearest_object = None

    # Find the nearest object the ray intersects
    for obj in objects:
        t = obj.intersect(ray)
        if t and t < nearest_t:
            nearest_t = t
            nearest_object = obj

    if nearest_object is None:
        return (0, 0, 0)  # Background color

    # Compute the intersection point
    intersection = ray.origin + ray.direction * nearest_t
    normal = (intersection - nearest_object.center).normalize()

    # Compute the lighting (simple diffuse shading)
    to_light = (light - intersection).normalize()
    intensity = max(normal.dot(to_light), 0)

    # Compute the color with shading
    r = min(int(nearest_object.color[0] * intensity), 255)
    g = min(int(nearest_object.color[1] * intensity), 255)
    b = min(int(nearest_object.color[2] * intensity), 255)

    return (r, g, b)

def render():
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    for i in range(WIDTH):
        for j in range(HEIGHT):
            # Compute the direction of the ray
            x = (2 * (i + 0.5) / WIDTH - 1) * np.tan(FOV / 2) * WIDTH / HEIGHT
            y = (1 - 2 * (j + 0.5) / HEIGHT) * np.tan(FOV / 2)
            direction = Vector(x, y, 1).normalize()
            ray = Ray(Vector(0, 0, 0), direction)

            color = cast_ray(ray, spheres, light)
            pixels[i, j] = color

    image.save("output.png")
    print("Rendering completed. Image saved as output.png")

if __name__ == "__main__":
    render()
