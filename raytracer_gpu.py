import numpy as np
import pyopencl as cl
from PIL import Image

# OpenCL kernel as a multi-line string
kernel_code = """
typedef struct {
    float x;
    float y;
    float z;
} Vector;

typedef struct {
    Vector origin;
    Vector direction;
} Ray;

typedef struct {
    Vector center;
    float radius;
    unsigned char color[3];
} Sphere;

// Ray-sphere intersection
float intersect_sphere(Ray ray, Sphere sphere) {
    Vector oc;
    oc.x = ray.origin.x - sphere.center.x;
    oc.y = ray.origin.y - sphere.center.y;
    oc.z = ray.origin.z - sphere.center.z;
    
    float a = ray.direction.x * ray.direction.x +
              ray.direction.y * ray.direction.y +
              ray.direction.z * ray.direction.z;
              
    float b = 2.0f * (oc.x * ray.direction.x +
                      oc.y * ray.direction.y +
                      oc.z * ray.direction.z);
                      
    float c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - sphere.radius * sphere.radius;
    
    float discriminant = b * b - 4.0f * a * c;
    
    if (discriminant < 0.0f) {
        return -1.0f; // No intersection
    } else {
        float sqrt_disc = sqrt(discriminant);
        float t1 = (-b - sqrt_disc) / (2.0f * a);
        float t2 = (-b + sqrt_disc) / (2.0f * a);
        if (t1 > 0.0f) {
            return t1;
        }
        if (t2 > 0.0f) {
            return t2;
        }
        return -1.0f;
    }
}

// Normalize a vector
Vector normalize(Vector v) {
    float norm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    Vector result;
    result.x = v.x / norm;
    result.y = v.y / norm;
    result.z = v.z / norm;
    return result;
}

// Dot product
float dot(Vector a, Vector b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Main kernel function
__kernel void render(
    __global const Sphere *spheres,
    const int num_spheres,
    __global const float *light_pos, // float3 as float array
    const float fov,
    const int width,
    const int height,
    __global unsigned char *image)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i >= width || j >= height) {
        return;
    }
    
    // Compute normalized device coordinates
    float x = (2.0f * ((float)i + 0.5f) / (float)width - 1.0f) * tan(fov / 2.0f) * ((float)width / (float)height);
    float y = (1.0f - 2.0f * ((float)j + 0.5f) / (float)height) * tan(fov / 2.0f);
    
    Vector dir;
    dir.x = x;
    dir.y = y;
    dir.z = 1.0f;
    dir = normalize(dir);
    
    Ray ray;
    ray.origin.x = 0.0f;
    ray.origin.y = 0.0f;
    ray.origin.z = 0.0f;
    ray.direction = dir;
    
    float nearest_t = 1e20f;
    int nearest_index = -1;
    
    // Find the nearest sphere intersected by the ray
    for (int k = 0; k < num_spheres; k++) {
        float t = intersect_sphere(ray, spheres[k]);
        if (t > 0.0f && t < nearest_t) {
            nearest_t = t;
            nearest_index = k;
        }
    }
    
    if (nearest_index == -1) {
        // Background color (black)
        image[(j * width + i) * 3 + 0] = 0;
        image[(j * width + i) * 3 + 1] = 0;
        image[(j * width + i) * 3 + 2] = 0;
        return;
    }
    
    // Compute intersection point
    Vector intersection;
    intersection.x = ray.origin.x + ray.direction.x * nearest_t;
    intersection.y = ray.origin.y + ray.direction.y * nearest_t;
    intersection.z = ray.origin.z + ray.direction.z * nearest_t;
    
    // Compute normal at the intersection
    Vector normal;
    normal.x = intersection.x - spheres[nearest_index].center.x;
    normal.y = intersection.y - spheres[nearest_index].center.y;
    normal.z = intersection.z - spheres[nearest_index].center.z;
    normal = normalize(normal);
    
    // Compute light direction
    Vector to_light;
    to_light.x = light_pos[0] - intersection.x;
    to_light.y = light_pos[1] - intersection.y;
    to_light.z = light_pos[2] - intersection.z;
    to_light = normalize(to_light);
    
    // Compute diffuse intensity
    float intensity = fmax(dot(normal, to_light), 0.0f);
    
    // Compute color with shading
    unsigned char r = (unsigned char)(fmin((float)spheres[nearest_index].color[0] * intensity, 255.0f));
    unsigned char g = (unsigned char)(fmin((float)spheres[nearest_index].color[1] * intensity, 255.0f));
    unsigned char b = (unsigned char)(fmin((float)spheres[nearest_index].color[2] * intensity, 255.0f));
    
    image[(j * width + i) * 3 + 0] = r;
    image[(j * width + i) * 3 + 1] = g;
    image[(j * width + i) * 3 + 2] = b;
}
"""

# Define Vector and Sphere as structured numpy dtypes
vector_dtype = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32)
])

sphere_dtype = np.dtype([
    ('center', vector_dtype),
    ('radius', np.float32),
    ('color', np.uint8, 3)
])

# Define spheres in the scene
spheres = np.array([
    ((0.0, -1.0, 3.0), 1.0, (255, 0, 0)),    # Red sphere
    ((2.0, 0.0, 4.0), 1.0, (0, 0, 255)),     # Blue sphere
    ((-2.0, 0.0, 4.0), 1.0, (0, 255, 0)),    # Green sphere
], dtype=sphere_dtype)

# Light position
light_pos = np.array([5.0, 5.0, -10.0], dtype=np.float32)

# Image dimensions and FOV
WIDTH = 800
HEIGHT = 600
FOV = np.pi / 3  # 60 degrees

# Initialize OpenCL
platforms = cl.get_platforms()
if not platforms:
    raise RuntimeError("No OpenCL platforms found.")

# Select the first platform
platform = platforms[0]

# Select the first GPU device; fall back to any device if GPU not found
devices = platform.get_devices(device_type=cl.device_type.GPU)
if not devices:
    devices = platform.get_devices(device_type=cl.device_type.ALL)
    if not devices:
        raise RuntimeError("No OpenCL devices found.")

device = devices[0]

# Create a context and command queue
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Compile the OpenCL kernel
program = cl.Program(context, kernel_code).build()

# Prepare image buffer
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Create OpenCL buffers
mf = cl.mem_flags
spheres_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=spheres)
image_buf = cl.Buffer(context, mf.WRITE_ONLY, image.nbytes)
light_pos_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=light_pos)

# Set kernel arguments and execute
program.render(
    queue,
    (WIDTH, HEIGHT),
    None,
    spheres_buf,
    np.int32(len(spheres)),
    light_pos_buf,
    np.float32(FOV),
    np.int32(WIDTH),
    np.int32(HEIGHT),
    image_buf
)

# Read the image buffer back to host
cl.enqueue_copy(queue, image, image_buf)
queue.finish()

# Reshape and save the image
image = image.reshape((HEIGHT, WIDTH, 3))
img = Image.fromarray(image, 'RGB')
img.save("output_gpu.png")
print("Rendering completed. Image saved as output_gpu.png")
